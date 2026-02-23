"""
Generate paired 2D Helmholtz datasets (damped and lossless) on ``[0,1]^2``.

Mathematical specification
--------------------------
For each sample, the code solves

``(-Delta - (1 - i*tan_delta) * kappa^2(x)) u(x) = s(x)``,

with ``kappa^2(x) = omega^2 / c(x)^2`` and a fixed source ``s`` given by a
centered Gaussian bump.

Sound-speed field:
1. Sample GRF ``g`` from inverse-DCT spectral synthesis:
   ``g = IDCT2(N * w ⊙ xi)``, ``xi_{k,l} ~ N(0,1)``,
   ``w_{k,l} = tau^(alpha-1) (pi^2(k^2+l^2)+tau^2)^(-alpha/2)``, ``w_{0,0}=0``.
2. Threshold map to binary speeds:
   ``c(x)=c_max`` if ``g(x)>=0``, else ``c(x)=c_min``.

Discretization:
- Uniform ``N x N`` finite-difference grid, spacing ``h = 1/(N-1)``.
- Operator ``-Delta`` assembled with 5-point stencil.
- Boundary handling:
  - ``bc="neumann"``: homogeneous Neumann via ghost-point elimination.
  - ``bc="robin"``: same Laplacian core plus boundary diagonal terms
    ``± 2 i gamma / h`` (sign depends on side, as implemented).

Outputs
-------
The script entrypoint writes:
- ``helmholtz_damped_dataset.pt``:
  - ``u`` shape ``(num_samples, 2, N, N)``, dtype ``torch.float32`` where
    channel 0 is real part and channel 1 is imaginary part.
  - ``c`` shape ``(num_samples, N, N)``, dtype ``torch.float32``.
- ``helmholtz_lossless_dataset.pt``: same structure, but solved with
  ``tan_delta = 0``.

RNG/seeding:
- ``make_datasets`` uses ``np.random.default_rng(seed)`` for all GRF draws.
- Within each sample, the same ``c`` is used for both damped and lossless
  solves.
"""

import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import idctn
from typing import Optional

# --- GRF sound speed (continuous, bounded) ---

def sample_grf(N: int, alpha: float = 2.0, tau: float = 3.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample an ``(N, N)`` GRF using inverse-DCT spectral filtering."""
    if rng is None:
        rng = np.random.default_rng()
    xi = rng.standard_normal((N, N))
    k = np.arange(N); K1, K2 = np.meshgrid(k, k, indexing='ij')
    spec = (tau**(alpha-1.0)) * (np.pi**2*(K1**2 + K2**2) + tau**2)**(-alpha/2.0)
    spec[0, 0] = 0.0
    g = idctn(N * spec * xi, type=2, norm='ortho')
    return g

def map_grf_to_speed(g: np.ndarray, c_min: float = 0.8, c_max: float = 1.2) -> np.ndarray:
    """Binary threshold map from GRF values to sound speed ``c(x)``."""
    return np.where(g >= 0, c_max, c_min)

# --- Fixed simple source (constant across all samples) ---

def fixed_source(N: int, sigma: float = 0.05, amplitude: float = 1.0) -> np.ndarray:
    """Return centered Gaussian source ``s`` on an ``N x N`` grid."""
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, N),
        np.linspace(0.0, 1.0, N),
        indexing="ij"
    )
    xc, yc = 0.5, 0.5
    r2 = (x - xc)**2 + (y - yc)**2
    return amplitude * np.exp(-r2 / (2.0 * sigma**2))

# --- Helmholtz assembly (Neumann or small Robin mismatch) ---

def _ij(i: int, j: int, N: int) -> int:
    return i * N + j

def assemble_helmholtz_neumann(kappa2: np.ndarray, dx: float) -> sp.csr_matrix:
    """
    Assemble FD matrix for ``(-Delta - kappa^2) u = s`` on ``[0,1]^2``
    with homogeneous Neumann BC (reflective) on all sides.

    Interior: standard 5-point stencil.
    Boundary: ghost-point elimination giving Neumann in the normal direction:
      - at left/right edges, the normal neighbor coefficient is doubled (-2/dx^2)
      - at bottom/top edges, same in y.
    """
    N = kappa2.shape[0]
    rows, cols, vals = [], [], []
    inv_dx2 = 1.0 / (dx * dx)

    for i in range(N):
        for j in range(N):
            c = _ij(i, j, N)

            # reaction term
            diag = -kappa2[i, j]

            # x-direction contribution to -Δ
            if 0 < i < N - 1:
                # interior in x: two neighbors
                rows.append(c); cols.append(_ij(i - 1, j, N)); vals.append(-inv_dx2)
                rows.append(c); cols.append(_ij(i + 1, j, N)); vals.append(-inv_dx2)
                diag += 2.0 * inv_dx2
            elif i == 0:
                # left boundary: Neumann via ghost elimination
                rows.append(c); cols.append(_ij(1, j, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2
            else:  # i == N - 1
                # right boundary
                rows.append(c); cols.append(_ij(N - 2, j, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2

            # y-direction contribution to -Δ
            if 0 < j < N - 1:
                # interior in y: two neighbors
                rows.append(c); cols.append(_ij(i, j - 1, N)); vals.append(-inv_dx2)
                rows.append(c); cols.append(_ij(i, j + 1, N)); vals.append(-inv_dx2)
                diag += 2.0 * inv_dx2
            elif j == 0:
                # bottom boundary
                rows.append(c); cols.append(_ij(i, 1, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2
            else:  # j == N - 1
                # top boundary
                rows.append(c); cols.append(_ij(i, N - 2, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2

            rows.append(c); cols.append(c); vals.append(diag)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N * N, N * N), dtype=np.complex128)


def assemble_helmholtz_robin(kappa2: np.ndarray, dx: float, gamma: float) -> sp.csr_matrix:
    """
    Assemble FD matrix for ``(-Delta - kappa^2) u = s`` on ``[0,1]^2``.

    The implemented boundary contribution adds side-dependent diagonal terms:
      left/bottom:  + (2 i γ / dx)
      right/top:    - (2 i γ / dx)

    together with the Neumann ghost-point elimination used in the Laplacian
    part. This is the exact discrete operator used by the solver.
    """
    N = kappa2.shape[0]
    rows, cols, vals = [], [], []
    inv_dx2 = 1.0 / (dx * dx)

    for i in range(N):
        for j in range(N):
            c = _ij(i, j, N)

            diag = -kappa2[i, j]

            # x-direction (-Δ) with Neumann ghost elimination
            if 0 < i < N - 1:
                # interior in x
                rows.append(c); cols.append(_ij(i - 1, j, N)); vals.append(-inv_dx2)
                rows.append(c); cols.append(_ij(i + 1, j, N)); vals.append(-inv_dx2)
                diag += 2.0 * inv_dx2
            elif i == 0:
                # left boundary
                rows.append(c); cols.append(_ij(1, j, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2
                # Robin term on left boundary
                diag += (2j * gamma) / dx
            else:  # i == N - 1
                # right boundary
                rows.append(c); cols.append(_ij(N - 2, j, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2
                # Robin term on right boundary
                diag += (-2j * gamma) / dx

            # y-direction (-Δ) with Neumann ghost elimination
            if 0 < j < N - 1:
                # interior in y
                rows.append(c); cols.append(_ij(i, j - 1, N)); vals.append(-inv_dx2)
                rows.append(c); cols.append(_ij(i, j + 1, N)); vals.append(-inv_dx2)
                diag += 2.0 * inv_dx2
            elif j == 0:
                # bottom boundary
                rows.append(c); cols.append(_ij(i, 1, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2
                # Robin term on bottom boundary
                diag += (2j * gamma) / dx
            else:  # j == N - 1
                # top boundary
                rows.append(c); cols.append(_ij(i, N - 2, N)); vals.append(-2.0 * inv_dx2)
                diag += 2.0 * inv_dx2
                # Robin term on top boundary
                diag += (-2j * gamma) / dx

            rows.append(c); cols.append(c); vals.append(diag)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N * N, N * N), dtype=np.complex128)


# --- Solve and dataset API ---

def solve_helmholtz_u(c: np.ndarray,
                      omega: float,
                      s: np.ndarray,
                      bc: str = "robin",
                      gamma: float = 0.003,
                      loss_tan: float = 0.0) -> np.ndarray:
    """
    Solve the discretized complex Helmholtz system for one sample.

    PDE form used:
    ``(-Delta - (1 - i*loss_tan) * omega^2 / c^2) u = s``.

    Parameters
    ----------
    c : np.ndarray
        Sound-speed field, shape ``(N, N)``.
    omega : float
        Angular frequency.
    s : np.ndarray
        Source field, shape ``(N, N)``.
    bc : str
        ``"neumann"`` or ``"robin"``.
    gamma : float
        Robin coefficient used only when ``bc="robin"``.
    loss_tan : float
        Uniform loss tangent; ``0`` yields lossless medium.

    Returns
    -------
    np.ndarray
        Complex solution ``u`` of shape ``(N, N)`` and dtype ``complex128``.
    """
    N = c.shape[0]
    dx = 1.0 / (N - 1)
    kappa2 = (omega**2) / (c**2)
    if loss_tan > 0.0:
        kappa2 = (1.0 - 1j*loss_tan) * kappa2  # uniform, fixed across all samples

    if bc == "neumann":
        A = assemble_helmholtz_neumann(kappa2, dx)
    elif bc == "robin":
        A = assemble_helmholtz_robin(kappa2, dx, gamma=gamma)
    else:
        raise ValueError("bc must be 'neumann' or 'robin'")

    u = spla.spsolve(A, s.reshape(-1).astype(np.complex128))
    return u.reshape(N, N)


def make_datasets(save_path_clean: str,
                  save_path_mismatch: str,
                  N: int = 64,
                  num_samples: int = 20000,
                  omega: float = 18.0,
                  c_min: float = 1.0,
                  c_max: float = 5.0,
                  alpha: float = 2.0,
                  tau: float = 3.0,
                  loss_tan: float = 0.02,
                  seed: int = 42,
                  bc='robin'):
    """
    Generate and save paired Helmholtz datasets with shared ``c`` realizations.

    For each sampled ``c``:
    - ``u_clean`` solves with ``loss_tan``.
    - ``u_mis`` solves with ``loss_tan=0``.

    Saved tensor layout for ``u`` is ``(num_samples, 2, N, N)``, where channel
    order is ``[real(u), imag(u)]``.
    """

    rng = np.random.default_rng(seed)
    s = fixed_source(N).astype(np.float64)

    u_clean, c_clean = [], []
    u_mis,   c_mis   = [], []


    for _ in range(num_samples):
        g = sample_grf(N, alpha=alpha, tau=tau, rng=rng)
        c = map_grf_to_speed(g, c_min=c_min, c_max=c_max)

        u = solve_helmholtz_u(c, omega=omega, s=s, bc=bc, gamma=0.03, loss_tan=loss_tan)
        u_clean.append(u); c_clean.append(c)

        u_r = solve_helmholtz_u(c, omega=omega, s=s, bc=bc, gamma=0.03, loss_tan=0.0)
        u_mis.append(u_r); c_mis.append(c)

    data_clean = {
        "u": torch.from_numpy(np.stack([np.stack([u.real, u.imag], axis=0) for u in u_clean])).to(torch.float32),
        "c": torch.from_numpy(np.stack(c_clean)).to(torch.float32),
    }
    data_mismatch = {
        "u": torch.from_numpy(np.stack([np.stack([u.real, u.imag], axis=0) for u in u_mis])).to(torch.float32),
        "c": torch.from_numpy(np.stack(c_mis)).to(torch.float32),
    }
    torch.save(data_clean,    save_path_clean)
    torch.save(data_mismatch, save_path_mismatch)


if __name__ == '__main__':

    path_damped = 'helmholtz_damped_dataset.pt'
    path_lossless = 'helmholtz_lossless_dataset.pt'

    make_datasets(path_damped, path_lossless, N=64, num_samples=10_000, omega=20.0, loss_tan=0.02,
                  alpha=4.0, tau=6.0,
                  c_min=0.8, c_max=1.2,
                  bc='robin')

