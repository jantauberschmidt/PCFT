"""
Generate a heterogeneous plane-strain elasticity dataset using PyTorch.

Mathematical specification
--------------------------
Domain: ``[0,1]^2`` discretized on an ``ny x nx`` Cartesian grid.

Material model:
- Scalar Young's modulus field ``E(y,x)`` sampled from a Voronoi tessellation.
- Isotropic linear elasticity in plane strain with fixed Poisson ratio ``nu``:
  ``mu = E/(2(1+nu))``,
  ``lambda = nu E / ((1+nu)(1-2nu))``.
- Strain/stress:
  ``eps = sym(grad u)``,
  ``sigma = 2 mu eps + lambda tr(eps) I``.

Solved equilibrium equation (no body force):
``div sigma = 0``.

Numerical scheme:
- First-order finite differences (central interior, one-sided at boundaries).
- Explicit pseudo-time relaxation on displacement:
  ``u^{k+1} = u^k + dt * div(sigma(u^k))``,
  with global conservative ``dt`` based on ``max(lambda+2mu)``.
- Iteration stops when
  ``sqrt(mean(rx^2 + ry^2)) <= tol``,
  where ``r = -div sigma`` as implemented.

Boundary conditions (Dirichlet):
- Left/right: fully clamped ``u=(0,0)``.
- Top: ``u_x=0``, ``u_y=A_top sin(pi x)``.
- Bottom:
  - if ``clamp_bottom=True``: ``u=(0,0)``;
  - else ``u_x=0``, ``u_y=-A_bottom sin(pi x)``.

Outputs
-------
The script entrypoint writes ``elasticity_dataset.pt`` with:
- ``E``: shape ``(N_samples, ny, nx)``, dtype ``torch.float32``.
- ``u``: shape ``(N_samples, ny, nx, 2)``, dtype ``torch.float32``,
  component order ``[..., 0]=u_x``, ``[..., 1]=u_y``.
- ``params``: Python dict with sampler and solver configuration.

RNG/seeding:
- Field sampling uses ``torch.rand``.
- ``__main__`` seeds PyTorch with ``SEED=42`` (and CUDA RNG if available).
"""

import math
import torch
import torch.nn.functional as F

# pick device
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))

# --------------------------
# Modulus sampling (Torch)
# --------------------------

@torch.no_grad()
def sample_modulus_voronoi_torch(batch_size, ny, nx, *,
                                 n_regions=48, low=0.5, high=5.0,
                                 blur_sigma=0.0, device=DEVICE, dtype=torch.float32):
    """
    Sample batched modulus fields ``E`` via Voronoi partitioning.

    Procedure per sample:
    1. Draw ``n_regions`` random sites uniformly over pixel coordinates.
    2. Assign each grid point to its nearest site (Euclidean distance).
    3. Draw per-region modulus values log-uniformly in ``[low, high]``.
    4. Optionally apply separable Gaussian blur (replicate padding).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, ny, nx)`` and dtype ``dtype`` on ``device``.
    """
    B = batch_size
    # grid
    yy = torch.linspace(0, ny - 1, ny, device=device, dtype=dtype).view(1, ny, 1)
    xx = torch.linspace(0, nx - 1, nx, device=device, dtype=dtype).view(1, 1, nx)

    # sites per sample (B, n_regions, 2)
    sites_y = torch.rand(B, n_regions, device=device, dtype=dtype) * (ny - 1)
    sites_x = torch.rand(B, n_regions, device=device, dtype=dtype) * (nx - 1)

    # region values per sample: log(E) is uniform on [log(low), log(high)]
    log_low = math.log(low); log_high = math.log(high)
    vals = torch.exp(torch.rand(B, n_regions, device=device, dtype=dtype) * (log_high - log_low) + log_low)

    # compute nearest site label via vectorised distances
    # d2: (B, n_regions, ny, nx)
    d2 = (yy - sites_y[:, :, None, None])**2 + (xx - sites_x[:, :, None, None])**2
    labels = torch.argmin(d2, dim=1)  # (B, ny, nx) indices in [0, n_regions)

    # gather per-cell E
    # convert (B, n_regions) -> (B, ny, nx) by take_along_dim
    E = torch.take_along_dim(vals, labels.view(B, -1), dim=1).view(B, ny, nx)

    if blur_sigma and blur_sigma > 0:
        radius = max(1, int(math.ceil(3 * blur_sigma)))
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        g1 = torch.exp(-0.5 * (x / blur_sigma)**2)
        g1 = (g1 / g1.sum()).view(1, 1, 1, -1)  # (out=1, in=1, 1, kx)

        # apply separable Gaussian blur via conv2d
        E = E.unsqueeze(1)  # (B,1,ny,nx)
        E = F.pad(E, (radius, radius, 0, 0), mode="replicate")
        E = F.conv2d(E, g1)
        E = F.pad(E, (0, 0, radius, radius), mode="replicate")
        E = F.conv2d(E, g1.transpose(-1, -2))
        E = E[:, 0]  # (B, ny, nx)

    return E


# --------------------------
# Batched solver (Torch)
# --------------------------

@torch.no_grad()
def solve_elasticity_batched(E, *, nu=0.30, max_iters=20000, tol=1e-6,
                             A_top=0.03, A_bottom=None, clamp_bottom=False):
    """
    Solve batched plane-strain linear elasticity by explicit relaxation.

    Parameters
    ----------
    E : torch.Tensor
        Young's modulus field(s), shape ``(B, ny, nx)``.
    nu : float
        Poisson ratio (constant across samples).
    max_iters : int
        Maximum relaxation iterations.
    tol : float
        Stop threshold for ``sqrt(mean(rx^2 + ry^2))`` with ``r=-div(sigma)``.
    A_top, A_bottom : float
        Vertical displacement amplitudes in top/bottom sine boundary profiles.
    clamp_bottom : bool
        If true, bottom boundary is fully clamped.

    Returns
    -------
    torch.Tensor
        Displacement tensor ``u`` of shape ``(B, ny, nx, 2)`` on ``DEVICE``.
    """
    if A_bottom is None:
        A_bottom = A_top

    E = E.to(DEVICE, dtype=torch.float32, non_blocking=True)
    B, ny, nx = E.shape
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    mu  = E / (2.0 * (1.0 + nu))
    lam = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam2mu = lam + 2.0 * mu

    # Boundary profiles are functions of x only and are broadcast over batch.
    x = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=torch.float32)
    u_top_y = +A_top * torch.sin(torch.pi * x)
    u_bot_y = -A_bottom * torch.sin(torch.pi * x)
    u_top_y = u_top_y.view(1, 1, nx).expand(B, 1, nx)    # (B,1,nx)
    if not clamp_bottom:
        u_bot_y = u_bot_y.view(1, 1, nx).expand(B, 1, nx)

    # unknown displacement
    u = torch.zeros((B, ny, nx, 2), device=DEVICE, dtype=torch.float32)

    # Conservative explicit step for the pseudo-time relaxation update.
    dt = 0.20 * (min(dx, dy)**2) / (float(torch.amax(lam2mu).item()) + 1e-12)

    def d_dx(a):
        # a: (B, ny, nx)
        out = torch.empty_like(a)
        out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2]) / (2 * dx)
        out[:, :, 0]    = (a[:, :, 1]  - a[:, :, 0])   / dx
        out[:, :, -1]   = (a[:, :, -1] - a[:, :, -2])  / dx
        return out

    def d_dy(a):
        out = torch.empty_like(a)
        out[:, 1:-1, :] = (a[:, 2:, :] - a[:, :-2, :]) / (2 * dy)
        out[:, 0,   :]  = (a[:, 1,   :] - a[:, 0,   :]) / dy
        out[:, -1,  :]  = (a[:, -1,  :] - a[:, -2,  :]) / dy
        return out

    def enforce_dirichlet(u_):
        # left/right fully clamped
        u_[:, :, 0, :]  = 0.0
        u_[:, :, -1, :] = 0.0
        # top: tangential fixed, normal prescribed
        u_[:, 0, :, 0] = 0.0
        u_[:, 0, :, 1] = u_top_y[:, 0, :]
        # bottom: either fully clamped or normal prescribed
        if clamp_bottom:
            u_[:, -1, :, :] = 0.0
        else:
            u_[:, -1, :, 0] = 0.0
            u_[:, -1, :, 1] = u_bot_y[:, 0, :]
        return u_

    for _ in range(max_iters):
        u = enforce_dirichlet(u)

        ux = u[..., 0]  # (B,ny,nx)
        uy = u[..., 1]

        dux_dx = d_dx(ux)
        duy_dy = d_dy(uy)
        dux_dy = d_dy(ux)
        duy_dx = d_dx(uy)

        exx = dux_dx
        eyy = duy_dy
        exy = 0.5 * (dux_dy + duy_dx)
        tr = exx + eyy

        sxx = 2.0 * mu * exx + lam * tr
        syy = 2.0 * mu * eyy + lam * tr
        sxy = 2.0 * mu * exy

        divx = d_dx(sxx) + d_dy(sxy)
        divy = d_dx(sxy) + d_dy(syy)

        rx = -divx
        ry = -divy
        rmean = torch.sqrt(torch.mean(rx*rx + ry*ry))
        if float(rmean.item()) <= tol:
            break

        u[..., 0] += dt * (-rx)
        u[..., 1] += dt * (-ry)


    u = enforce_dirichlet(u)
    return u


# --------------------------
# Fast dataset generation
# --------------------------

def generate_dataset_fast(N_samples, ny, nx, *,
                          batch_size=64,
                          n_regions=32, low=1.0, high=10.0, blur_sigma=1.0,
                          nu=0.30, max_iters=20_000, tol=1e-6,
                          A_top=0.10, A_bottom=None, clamp_bottom=False,
                          device=DEVICE):
    """
    Generate dataset in batches and return CPU tensors ready for ``torch.save``.

    Returned dictionary:
    - ``E``: ``(N_samples, ny, nx)``, ``float32``.
    - ``u``: ``(N_samples, ny, nx, 2)``, ``float32``.
    - ``params``: metadata dict describing sampler/solver settings.
    """
    E_batches = []
    U_batches = []

    remaining = N_samples
    while remaining > 0:
        b = min(batch_size, remaining)

        # sample E on device
        E_b = sample_modulus_voronoi_torch(b, ny, nx,
                                           n_regions=n_regions, low=low, high=high,
                                           blur_sigma=blur_sigma, device=device)

        # solve on device
        U_b = solve_elasticity_batched(E_b, nu=nu, max_iters=max_iters, tol=tol,
                                       A_top=A_top, A_bottom=A_bottom, clamp_bottom=clamp_bottom)

        # move to CPU for storage (float32)
        E_batches.append(E_b.detach().cpu())
        U_batches.append(U_b.detach().cpu())

        remaining -= b

    E = torch.cat(E_batches, dim=0)              # (N, ny, nx)
    U = torch.cat(U_batches, dim=0)              # (N, ny, nx, 2)

    params = {
        'ny': ny, 'nx': nx, 'N_samples': N_samples,
        'sampler': {'type': 'voronoi', 'n_regions': n_regions, 'low': low, 'high': high, 'blur_sigma': blur_sigma},
        'solver': {'fn': 'solve_elasticity_batched', 'nu': nu, 'max_iters': max_iters, 'tol': tol,
                   'A_top': A_top, 'A_bottom': (A_top if A_bottom is None else A_bottom),
                   'clamp_bottom': clamp_bottom, 'device': str(device)},
    }
    return {'E': E, 'u': U, 'params': params}


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ny, nx = 64, 64
    N = 10_000

    data = generate_dataset_fast(
        N, ny, nx,
        batch_size=128,
        n_regions=32, low=1.0, high=10.0, blur_sigma=1.0,
        nu=0.30, max_iters=20_000, tol=1e-6,
        A_top=0.10, A_bottom=None, clamp_bottom=False,
        device=DEVICE,
    )

    save_path = "elasticity_dataset.pt"
    torch.save(data, save_path)
