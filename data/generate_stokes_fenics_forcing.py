"""
Generate steady Stokes datasets with spatially varying viscosity using FEniCS.

Mathematical specification
--------------------------
Domain: ``Omega = [0,1]^2`` with an ``N x N`` sampling grid for outputs, and a
FEniCS ``UnitSquareMesh(N-1, N-1)`` for the FEM solve.

For each sample, viscosity is generated as:
1. ``g = IDCT2(N * w ⊙ xi)``, ``xi_{k,l} ~ N(0,1)``,
   ``w_{k,l} = tau^(alpha-1) (pi^2(k^2+l^2)+tau^2)^(-alpha/2)``, ``w_{0,0}=0``.
2. ``nu(x)=nu_max`` where ``g(x)>=0``, else ``nu_min``.

Implemented PDE system:
``-div(nu(x) grad u) + grad p = f``, ``div u = 0``.
Weak form assembled (Taylor-Hood ``P2/P1``):
``∫ nu grad(u):grad(v) dx - ∫ p div(v) dx - ∫ q div(u) dx = ∫ f·v dx``.

Boundary conditions:
- Top wall ``y=1``: ``u = (U_lid sin^2(pi x), 0)``.
- Bottom/left/right walls: ``u = (0,0)``.

Body force for dataset family:
``f(x,y) = (F0 sin(pi y), 0)`` (including ``F0=0`` case).

Outputs
-------
For each forcing/suffix pair, saves ``{base_name}{suffix}.pt`` with:
- ``u``: shape ``(num_samples, 3, N, N)``, dtype ``torch.float32``,
  channel order ``[u_x, u_y, p]`` with per-sample mean-zero pressure.
- ``nu``: shape ``(num_samples, N, N)``, dtype ``torch.float32``.

RNG/seeding:
- GRFs are sampled with ``np.random.default_rng(seed)``.
- In ``create_forcing_datasets``, RNG is re-initialized for each forcing value,
  so each forcing dataset uses the same viscosity draw sequence (up to rejected
  samples due to solver/validity filtering).
"""

import numpy as np
import torch
from scipy.fft import idctn
from typing import Optional
from math import sin, pi

# FEniCS (classic)
from dolfin import (
    UnitSquareMesh,
    VectorElement,
    FiniteElement,
    MixedElement,
    FunctionSpace,
    TrialFunctions,
    TestFunctions,
    Function,
    Constant,
    DirichletBC,
    SubDomain,
    UserExpression,
    inner,
    grad,
    div,
    dx,
    solve,
    Point,
)


# --- GRF viscosity (smooth mapping, all variation from nu) ---

def sample_grf(
    N: int,
    alpha: float = 2.0,
    tau: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a GRF on an ``N x N`` grid using inverse-DCT spectral synthesis."""
    if rng is None:
        rng = np.random.default_rng()
    xi = rng.standard_normal((N, N))
    k = np.arange(N)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    spec = (tau**(alpha - 1.0)) * (np.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha / 2.0)
    spec[0, 0] = 0.0
    g = idctn(N * spec * xi, type=2, norm="ortho")
    return g

def map_grf_to_viscosity(
    g: np.ndarray,
    nu_min: float = 0.05,
    nu_max: float = 0.20,
) -> np.ndarray:
    """Threshold-map GRF values to two-level viscosity ``nu(x)``."""
    return np.where(g >= 0.0, nu_max, nu_min)

# --- FEniCS-side: lid-driven cavity domains ---

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.0) < 1e-14


class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < 1e-14


class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < 1e-14


class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.0) < 1e-14


class LidVelocityExpression(UserExpression):
    """
    Smooth lid profile:
        u_x(x, y=1) = U_lid * sin^2(pi * x),
        u_y(x, y=1) = 0.
    Vanishes with zero derivative at x=0,1, maximum at x=0.5.
    """
    def __init__(self, U_lid, **kwargs):
        self.U_lid = float(U_lid)
        super().__init__(**kwargs)

    def eval(self, values, x):
        s = x[0]
        phi = sin(pi * s) ** 2
        values[0] = self.U_lid * phi
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class ForcingExpression(UserExpression):
    """
    Kolmogorov-type body forcing:
        f(x, y) = F0 * (sin(pi * y), 0).
    Divergence-free, nonzero curl.
    """
    def __init__(self, F0, **kwargs):
        self.F0 = float(F0)
        super().__init__(**kwargs)

    def eval(self, values, x):
        y = x[1]
        values[0] = self.F0 * np.sin(np.pi * y)
        values[1] = 0.0

    def value_shape(self):
        return (2,)

# --- Steady Stokes solver with optional forcing (Taylor–Hood) ---

def solve_stokes_lid_driven_fenics(
    nu_grid: np.ndarray,
    U_lid: float = 1.0,
    N: int = 64,
    pdeg_u: int = 2,
    pdeg_p: int = 1,
    with_forcing: bool = False,
    F0: float = 0.0,
) -> np.ndarray:
    """
    Solve steady Stokes with variable viscosity ``nu(x)``:

        -div(nu(x) grad u) + grad p = f(x),
        div u = 0

    on [0,1]^2 with smooth lid-driven cavity BC:
        top (y=1):    u = (U_lid * sin^2(pi x), 0),
        other walls:  u = (0, 0).

    If with_forcing == False:
        f(x) = 0  (pure lid-driven Stokes).

    If with_forcing == True:
        f(x, y) = F0 * (sin(pi y), 0) (Kolmogorov-type forcing).

    nu_grid is N x N, defined on a uniform grid over [0,1]^2
    with indices (i,j) ~ (y,x), as in your other datasets.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, N, N)`` with channels ``(u_x, u_y, p)``
        sampled on the uniform output grid, with pressure shifted to mean zero.
    """
    # FE mesh: (N-1)x(N-1) squares, each split into 2 triangles
    mesh = UnitSquareMesh(N - 1, N - 1)

    # Taylor–Hood: P2 vector for u, P1 scalar for p
    Ve = VectorElement("P", mesh.ufl_cell(), pdeg_u)
    Qe = FiniteElement("P", mesh.ufl_cell(), pdeg_p)
    W = FunctionSpace(mesh, MixedElement([Ve, Qe]))

    # Trial/test
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Encode nu(x) as a UserExpression that samples nu_grid
    class NuExpression(UserExpression):
        def __init__(self, nu_grid, N, **kwargs):
            self.nu_grid = nu_grid
            self.N = N
            super().__init__(**kwargs)

        def eval(self, values, x):
            # x[0] = x, x[1] = y in [0,1]
            j = int(round(x[0] * (self.N - 1)))
            i = int(round(x[1] * (self.N - 1)))

            # clamp indices
            if i < 0:
                i = 0
            elif i > self.N - 1:
                i = self.N - 1
            if j < 0:
                j = 0
            elif j > self.N - 1:
                j = self.N - 1

            values[0] = float(self.nu_grid[i, j])

        def value_shape(self):
            return ()

    nu_expr = NuExpression(nu_grid, N, degree=0)

    # Weak form assembled exactly as:
    #   ∫ nu grad(u):grad(v) dx - ∫ p div(v) dx - ∫ q div(u) dx = ∫ f·v dx.
    a = inner(nu_expr * grad(u), grad(v)) * dx \
        - div(v) * p * dx \
        - q * div(u) * dx

    if with_forcing:
        f_expr = ForcingExpression(F0, degree=2)
        L = inner(f_expr, v) * dx
    else:
        # Zero body force branch.
        L = Constant(0.0) * q * dx

    top = TopBoundary()
    bottom = BottomBoundary()
    left = LeftBoundary()
    right = RightBoundary()

    lid_expr = LidVelocityExpression(U_lid, degree=2)
    bc_top = DirichletBC(W.sub(0), lid_expr, top)
    bc_bot = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bottom)
    bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)
    bc_right = DirichletBC(W.sub(0), Constant((0.0, 0.0)), right)
    bcs = [bc_top, bc_bot, bc_left, bc_right]

    # Solve linear system
    w = Function(W)
    solve(a == L, w, bcs)

    u_fe, p_fe = w.split(deepcopy=True)

    # Sample solution on uniform N x N grid
    u_x = np.zeros((N, N), dtype=np.float64)
    u_y = np.zeros((N, N), dtype=np.float64)
    p_arr = np.zeros((N, N), dtype=np.float64)

    xs = np.linspace(0.0, 1.0, N)
    ys = np.linspace(0.0, 1.0, N)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            point = Point(x, y)
            ux_val, uy_val = u_fe(point)
            p_val = p_fe(point)
            u_x[i, j] = ux_val
            u_y[i, j] = uy_val
            p_arr[i, j] = p_val

    # fix pressure gauge: mean-zero pressure per sample
    p_arr = p_arr - p_arr.mean()

    return np.stack([u_x, u_y, p_arr], axis=0)


# --- Dataset API (same footprint, now: forced vs unforced) ---

def is_solution_valid(
    up: np.ndarray,
    U_lid: float,
    max_u_factor: float = 5.0,
    max_p_abs: float = 50.0,
) -> bool:
    """
    up: (3, N, N) array with (u_x, u_y, p).
    Basic acceptance filter used during dataset generation:
    - finite values only,
    - ``max ||u|| <= max_u_factor * U_lid``,
    - ``max |p| <= max_p_abs``.
    """
    if not np.isfinite(up).all():
        return False

    u_x, u_y, p = up
    max_u = float(np.max(np.sqrt(u_x**2 + u_y**2)))
    max_p = float(np.max(np.abs(p)))

    if max_u > max_u_factor * U_lid:
        return False

    if max_p > max_p_abs:
        return False

    return True

def create_forcing_datasets(
    base_name: str,
    forcings = None,
    suffixes = None,
) -> None:
    """
    Generate one dataset file per forcing amplitude in ``forcings``.

    Each saved file contains only forced-solve outputs (for the specified ``F0``)
    and matching viscosity fields.
    """

    N = 64
    num_samples = 256
    nu_min = 0.02
    nu_max = 0.5
    alpha = 4.0
    tau = 6.0
    U_lid = 1.0
    seed = 42

    if forcings is None:
        forcings = [0.0, 0.5, 1.0, 1.5, 2.0]
        suffixes = ['_0p0', '_0p5', '_1p0', '_1p5', '_2p0']

    elif suffixes is None:
        raise ValueError('If forcings is specified, suffixes must be provided.')


    for F0, ending in zip(forcings, suffixes):

        rng = np.random.default_rng(seed)

        u_forced_list = []
        nu_list = []

        num_valid = 0
        num_attempts = 0
        max_attempts = 10 * num_samples  # safety

        while num_valid < num_samples and num_attempts < max_attempts:
            num_attempts += 1

            g = sample_grf(N, alpha=alpha, tau=tau, rng=rng)
            nu = map_grf_to_viscosity(g, nu_min=nu_min, nu_max=nu_max)

            try:
                up_forced = solve_stokes_lid_driven_fenics(
                    nu_grid=nu,
                    U_lid=U_lid,
                    N=N,
                    with_forcing=True,
                    F0=F0,
                )
            except RuntimeError:
                # FEniCS solver might throw in truly bad cases
                continue

            # Accept sample only if the solved field passes sanity checks.
            if not is_solution_valid(up_forced, U_lid=U_lid):
                continue

            u_forced_list.append(up_forced.astype(np.float64))
            nu_list.append(nu.astype(np.float64))

            num_valid += 1

        if num_valid < num_samples:
            raise RuntimeError(
                f"Only generated {num_valid} valid samples out of {num_samples} requested "
                f"after {num_attempts} attempts."
            )

        u_forced_arr = np.stack(u_forced_list, axis=0)
        nu_arr = np.stack(nu_list, axis=0)

        data_forced = {
            "u": torch.from_numpy(u_forced_arr).to(torch.float32),
            "nu": torch.from_numpy(nu_arr).to(torch.float32),
        }
        save_name = base_name + ending + '.pt'
        torch.save(data_forced, save_name)

        print('')
        print('')
        print('')
        print('')
        print(f'finished dataset {save_name}')
        print('')
        print('')
        print('')
        print('')


if __name__ == "__main__":
    create_forcing_datasets(
        base_name='stokes_forced',
        forcings=[0.0, 0.5, 1.0, 1.5, 2.0],
        suffixes = ['_0p0', '_0p5', '_1p0', '_1p5', '_2p0'],
        )
