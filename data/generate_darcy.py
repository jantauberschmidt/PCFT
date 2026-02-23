"""
Generate a Darcy-flow dataset on a uniform square grid.

Mathematical specification
--------------------------
Domain: unit square ``[0,1]^2`` on an ``N x N`` Cartesian grid with spacing
``h = 1/(N-1)``.

Random coefficient generation:
1. Draw ``xi_{k,l} ~ N(0,1)`` i.i.d. for ``k,l = 0,...,N-1``.
2. Form spectral weights
   ``w_{k,l} = tau^(alpha-1) * (pi^2 (k^2 + l^2) + tau^2)^(-alpha/2)`` with
   ``w_{0,0}=0``.
3. Set ``a_raw = IDCT2(N * w ⊙ xi)`` (type-II, orthonormal).
4. Map to piecewise-constant permeability:
   ``a(x)=12`` if ``a_raw(x) >= 0`` and ``a(x)=3`` otherwise.

PDE solved per sample:
``-div(a grad u) = f`` with ``f(x)=1`` and homogeneous Dirichlet boundary
conditions ``u=0`` on ``∂[0,1]^2``. The implementation uses a 5-point,
flux-form finite-difference discretization with arithmetic face averaging and
solves the sparse linear system using ``scipy.sparse.linalg.spsolve``.

Outputs
-------
When run as a script, writes ``darcy_dataset.pt`` containing:
- ``data["a"]``: permeability tensor, shape ``(num_samples, N, N)``,
  dtype ``torch.float32``.
- ``data["u"]``: solution tensor, shape ``(num_samples, N, N)``,
  dtype ``torch.float32``.

RNG/seeding:
- ``sample_grf`` uses NumPy global RNG (``np.random.randn``).
- ``__main__`` seeds Python ``random``, NumPy, and PyTorch with ``SEED=42``.
"""

import torch
import torch.fft
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import idctn
import numpy as np
import random


def sample_grf(N, alpha=2.0, tau=3):
    """
    Sample a 2D Gaussian random field via inverse DCT spectral synthesis.

    Parameters
    ----------
    N : int
        Grid size (returns ``N x N`` field).
    alpha : float
        Spectral smoothness exponent in
        ``(pi^2 (k^2+l^2)+tau^2)^(-alpha/2)``.
    tau : float
        Correlation-scale parameter in the same spectrum.

    Returns
    -------
    torch.Tensor
        Field ``a_raw`` with shape ``(N, N)`` and dtype ``torch.float32``.

    Notes
    -----
    Uses NumPy global RNG via ``np.random.randn``; reproducibility therefore
    depends on external ``np.random.seed(...)`` calls.
    """
    xi = np.random.randn(N, N)
    k1 = np.arange(N)
    k2 = np.arange(N)
    K1, K2 = np.meshgrid(k1, k2, indexing='ij')
    coef = (tau ** (alpha - 1)) * (np.pi ** 2 * (K1 ** 2 + K2 ** 2) + tau ** 2) ** (-alpha / 2)
    coef[0, 0] = 0.0  # remove constant mode
    L = N * coef * xi
    a_raw = idctn(L, type=2, norm='ortho')
    return torch.tensor(a_raw, dtype=torch.float32)


def psi(a_raw):
    """Map a raw GRF realization to two-phase permeability values.

    ``a = 12`` where ``a_raw >= 0`` and ``a = 3`` otherwise.
    """
    return torch.where(a_raw >= 0, torch.tensor(12.0, dtype=torch.float32),
                                 torch.tensor(3.0,  dtype=torch.float32))


def solve_pde(a: torch.Tensor, f: torch.Tensor):
    """
    Solve the discrete variable-coefficient Poisson/Darcy system on ``[0,1]^2``.

    Implemented strong form:
    ``-div(a grad u) = f`` with ``u=0`` on the boundary.

    Discretization:
    - Uniform grid with ``h = 1/(N-1)``.
    - 5-point flux form using arithmetic face averages, e.g.
      ``a_{i+1/2,j} = 0.5 * (a_{i,j} + a_{i+1,j})``.
    - Interior equation assembled as
      ``A u = -f`` (equivalently the standard ``-div(a grad u)=f`` form).

    Parameters
    ----------
    a : torch.Tensor
        Permeability field, shape ``(N, N)``, dtype float.
    f : torch.Tensor
        RHS field, shape ``(N, N)``, dtype float.

    Returns
    -------
    torch.Tensor
        Solution ``u`` with shape ``(N, N)`` and dtype ``torch.float32``.
    """
    N = a.shape[0]
    dx = 1.0 / (N - 1)
    dx2 = dx * dx

    a = a.numpy()
    f = f.numpy()

    A = scipy.sparse.lil_matrix((N * N, N * N))
    # Interior stencil is assembled for the discrete divergence operator, so
    # the linear system is written as A u = -f.
    b = -f.reshape(-1)

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                # Dirichlet BC: u = 0
                A[idx, idx] = 1.0
                b[idx] = 0.0
            else:
                ai_j = a[i, j]
                a_ip1 = 0.5 * (ai_j + a[i + 1, j])
                a_im1 = 0.5 * (ai_j + a[i - 1, j])
                a_jp1 = 0.5 * (ai_j + a[i, j + 1])
                a_jm1 = 0.5 * (ai_j + a[i, j - 1])

                A[idx, idx]       = -(a_ip1 + a_im1 + a_jp1 + a_jm1) / dx2
                A[idx, idx + 1]   = a_jp1 / dx2
                A[idx, idx - 1]   = a_jm1 / dx2
                A[idx, idx + N]   = a_ip1 / dx2
                A[idx, idx - N]   = a_im1 / dx2

    A = A.tocsr()
    u = scipy.sparse.linalg.spsolve(A, b)
    u = torch.tensor(u.reshape(N, N), dtype=torch.float32)
    return u


def generate_sample(N):
    a_raw = sample_grf(N, alpha=2.0, tau=3.0)
    a = psi(a_raw)
    f = torch.ones(N, N, dtype=torch.float32)  # constant forcing
    u = solve_pde(a, f)
    return a, u


def forward_solve():
    N = 64
    f = torch.ones(N, N, dtype=torch.float32)

    def solve(a):
        return solve_pde(a, f)

    return solve

if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # ---------------------------------------------------------

    N = 64  # grid size
    num_samples = 20_000

    a_list = []
    u_list = []

    for _ in range(num_samples):
        a, u = generate_sample(N)
        a_list.append(a)
        u_list.append(u)

    data = {
        'a': torch.stack(a_list),  # shape: (num_samples, N, N)
        'u': torch.stack(u_list),
    }

    torch.save(data, 'darcy_dataset.pt')
