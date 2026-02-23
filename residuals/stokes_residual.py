"""
Steady Stokes residuals for velocity-pressure fields on 2-D grids.

Mathematical specification
--------------------------
Target PDE used by this code:
``-div(nu grad u) + grad p = f``,
``div u = 0``,
with velocity ``u=(u_x,u_y)`` and optional Kolmogorov forcing
``f=(F0 sin(pi y), 0)``.

Implemented residuals:
1. Strong residual (`compute_strong_stokes_residual`):
   finite-difference evaluation of
   ``r_x = -div(nu grad u_x) + d_x p - f_x``,
   ``r_y = -div(nu grad u_y) + d_y p - f_y``,
   ``r_div = d_x u_x + d_y u_y``.
   Returns
   ``sqrt(int(r_x^2+r_y^2+r_div^2))/sqrt(int nu|grad u|^2 + eps)``.
2. Weak residual (`compute_weak_stokes_residual`):
   patchwise test moments
   ``R_x,n = int nu grad u_x·grad psi_n + int d_x p psi_n - int f_x psi_n``,
   ``R_y,n = int nu grad u_y·grad psi_n + int d_y p psi_n - int f_y psi_n``,
   ``R_div,n = int (d_x u_x + d_y u_y) psi_n``.
   Uses `prepare_test_functions_nd` for test family and trapezoidal quadrature.
   Each residual is normalized by ``sqrt(int nu|grad u|^2 + 1e-8)`` per patch.

Inputs/Outputs
--------------
- State channel order is ``[u_x, u_y, p]`` with shape ``(B,3,H,W)``.
- Viscosity `nu` can be ``(B,1,H,W)`` or ``(B,H,W)``.
- Weak residual function returns ``(B,3,N_test)`` plus test centers.
- Wrapper `WeakStokesResidual.compute_residual` returns shape ``(B,)`` after
  squared aggregation and global scaling.

"""

import torch
import math
from typing import Sequence, Tuple

from residuals.common import prepare_test_functions_nd

def _stokes_kolmogorov_forcing(
    F0: float,
    H: int,
    W: int,
    ranges: Sequence[Tuple[float, float]],
    device: torch.device,
) -> torch.Tensor:
    """
    Return forcing channels ``(f_x, f_y)`` on a uniform ``H x W`` grid.

    Implemented formula: ``f_x = F0 * sin(pi * y)``, ``f_y = 0``.
    """
    if F0 == 0.0:
        return torch.zeros((2, H, W), device=device, dtype=torch.float32)
    (y0, y1), _ = ranges
    y = torch.linspace(y0, y1, H, device=device, dtype=torch.float32)  # shape (H,)
    fy = torch.zeros((H, W), device=device, dtype=torch.float32)
    fx_line = F0 * torch.sin(torch.pi * y)                              # shape (H,)
    fx = fx_line.unsqueeze(1).expand(H, W)                              # (H,W)
    return torch.stack([fx, fy], dim=0)


def compute_strong_stokes_residual(
    u: torch.Tensor,
    nu: torch.Tensor,
    ranges: Sequence[Tuple[float, float]] = None,
    F0: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    Strong-form residual for 2-D steady Stokes with variable viscosity $\nu(x)$,
    discretized with `torch.gradient` (second-order central differences in the
    interior, one-sided at the boundary):

        -\nabla\!\cdot(\nu \nabla u) + \nabla p = f, \qquad \nabla\!\cdot u = 0,

    where the (optional) body force is Kolmogorov type
        $f(x,y) = F_0(\sin(\pi y), 0)$.

    Pointwise residual components:
        r_x   = -div(\nu ∇u_x) + ∂_x p - f_x,
        r_y   = -div(\nu ∇u_y) + ∂_y p - f_y,
        r_div = ∂_x u_x + ∂_y u_y.

    The returned scalar per sample is the energy-normalized L2 residual
        R_strong = ||(r_x,r_y,r_div)||_{L^2(Ω)} / sqrt(E(u) + eps),
    with
        E(u) = ∫_Ω \nu |\nabla u|^2 dx
    (matching the weak residual with lam = 0.0).
    """
    if ranges is None:
        ranges = [(0.0, 1.0), (0.0, 1.0)]

    assert u.ndim == 4 and u.size(1) == 3, "u must be (B,3,H,W)"
    B, _, H, W = u.shape
    device = u.device

    dy, dx = [(r[1] - r[0]) / (n - 1) for r, n in zip(ranges, (H, W))]
    dA = dx * dy

    if nu.ndim == 4 and nu.size(1) == 1:
        nu_s = nu[:, 0]
    elif nu.ndim == 3:
        nu_s = nu
    else:
        raise ValueError("nu must be [B,1,H,W] or [B,H,W].")

    u_x = u[:, 0]
    u_y = u[:, 1]
    p   = u[:, 2]

    # First derivatives
    dux_dy, dux_dx = torch.gradient(u_x, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    duy_dy, duy_dx = torch.gradient(u_y, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dp_dy,  dp_dx  = torch.gradient(p,   spacing=(dy, dx), dim=(1, 2), edge_order=2)

    # Fluxes q = nu ∇u and their divergence
    qx_x = nu_s * dux_dx
    qx_y = nu_s * dux_dy
    qy_x = nu_s * duy_dx
    qy_y = nu_s * duy_dy

    dqx_x_dy, dqx_x_dx = torch.gradient(qx_x, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dqx_y_dy, dqx_y_dx = torch.gradient(qx_y, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    div_nu_grad_ux = dqx_x_dx + dqx_y_dy

    dqy_x_dy, dqy_x_dx = torch.gradient(qy_x, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dqy_y_dy, dqy_y_dx = torch.gradient(qy_y, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    div_nu_grad_uy = dqy_x_dx + dqy_y_dy

    # Forcing (broadcast to batch)
    f = _stokes_kolmogorov_forcing(F0, H, W, ranges, device)       # (H,W)
    fx = f[0].unsqueeze(0).expand(B, -1, -1)                               # (B,H,W)
    fy = f[1].unsqueeze(0).expand(B, -1, -1)

    # Strong residual components
    r_x   = -div_nu_grad_ux + dp_dx - fx
    r_y   = -div_nu_grad_uy + dp_dy - fy
    r_div = dux_dx + duy_dy

    res_sq = ((r_x**2 + r_y**2 + r_div**2).sum(dim=(1, 2))) * dA
    res_L2 = torch.sqrt(res_sq + eps)

    grad2 = dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2
    grad_energy = (nu_s * grad2).sum(dim=(1, 2)) * dA

    R_strong = res_L2 / (torch.sqrt(grad_energy + eps) + eps)
    return R_strong



def compute_weak_stokes_residual(
    u: torch.Tensor,
    nu: torch.Tensor,
    ranges,                       # [(x0,x1),(y0,y1)]
    sigma_range=(5.0, 20.0),
    test_fun: str = "wd_wv",
    F0: float = 0.0,
    N_test=None,
):
    """
    Compute weak residual moments for steady Stokes momentum + incompressibility.

    No Brinkman term is included (the internal ``lam`` is fixed to 0).

    Returns
    -------
    residual : torch.Tensor
        Shape ``(B,3,N_test)`` for ``[x-momentum, y-momentum, divergence]``.
    center_coords : list[torch.Tensor]
        Test-function centers from ``prepare_test_functions_nd``.
    """
    assert u.ndim == 4 and u.size(1) == 3, "u must be [B,3,H,W]"
    B, _, H, W = u.shape
    device = u.device

    # viscosity to [B,H,W]
    if nu.ndim == 4 and nu.size(1) == 1:
        nu_s = nu[:, 0]
    elif nu.ndim == 3:
        nu_s = nu
    else:
        raise ValueError("nu must be [B,1,H,W] or [B,H,W].")

    # fold channels into batch: [B*3,H,W]
    u_bc  = u.reshape(B * 3, H, W)
    nu_bc = nu_s.repeat_interleave(3, dim=0)  # [B*3,H,W]

    f = _stokes_kolmogorov_forcing(F0, H, W, ranges, device)       # (2,H,W)

    (u_patch, nu_patch, f_patch, grad_u_patch, phi_moll,
     grad_phi_moll, center_coords, weights_patch, dA) = prepare_test_functions_nd(
        u_bc, nu_bc, ranges,
        source=f,
        device=device,
        sigma_range=sigma_range,
        test_fun=test_fun,
        N_test=N_test,
    )
    # u_patch      : [BC, N, Ky, Kx]
    # nu_patch     : [BC, N, Ky, Kx]
    # f_patch     : [2, N, Ky, Kx]
    # grad_u_patch : [BC, N, Ky, Kx, 2] (∂y, ∂x)
    # phi_moll     : [N,Ky,Kx]
    # grad_phi_moll: [N,Ky,Kx,2]
    # weights_patch: [N,Ky,Kx]

    N = grad_u_patch.shape[1]
    Ky, Kx = grad_u_patch.shape[2], grad_u_patch.shape[3]

    # unfold component dimension: [B,3,N,Ky,Kx,2]
    grad_u_patch = grad_u_patch.view(B, 3, N, Ky, Kx, 2)
    u_patch      = u_patch.view(B, 3, N, Ky, Kx)
    nu_patch     = nu_patch.view(B, 3, N, Ky, Kx)[:, 0]  # same nu for all channels -> [B,N,Ky,Kx]
    f_patch_b = f_patch.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B,2,N,Ky,Kx]
    fx_patch = f_patch_b[:, 0]  # [B,N,Ky,Kx]
    fy_patch = f_patch_b[:, 1]  # [B,N,Ky,Kx]

    # gradient components: [B,N,Ky,Kx]
    dux_dy = grad_u_patch[:, 0, ..., 0]
    dux_dx = grad_u_patch[:, 0, ..., 1]
    duy_dy = grad_u_patch[:, 1, ..., 0]
    duy_dx = grad_u_patch[:, 1, ..., 1]
    dp_dy  = grad_u_patch[:, 2, ..., 0]
    dp_dx  = grad_u_patch[:, 2, ..., 1]

    # u on patches: [B,N,Ky,Kx]
    u_x_patch = u_patch[:, 0]
    u_y_patch = u_patch[:, 1]

    # test function & weights
    gpy = grad_phi_moll[..., 0]  # [N,Ky,Kx] ∂y φ
    gpx = grad_phi_moll[..., 1]  # [N,Ky,Kx] ∂x φ
    phi = phi_moll               # [N,Ky,Kx]
    W   = weights_patch          # [N,Ky,Kx]

    def integ(t: torch.Tensor) -> torch.Tensor:
        # t: [B,N,Ky,Kx]
        return (t * W * dA).sum(dim=(-2, -1))

    # grad terms: ∫ nu ∇u · ∇φ
    grad_term_x = integ(nu_patch * (dux_dy * gpy + dux_dx * gpx))  # [B,N]
    grad_term_y = integ(nu_patch * (duy_dy * gpy + duy_dx * gpx))

    f_term_x = integ(fx_patch * phi)  # [B,N]
    f_term_y = integ(fy_patch * phi)  # [B,N]


    # pressure gradient terms: ∫ (∂_x p) φ, ∫ (∂_y p) φ
    p_term_x = integ(dp_dx * phi)
    p_term_y = integ(dp_dy * phi)

    # divergence term: ∫ (∂_x u_x + ∂_y u_y) φ
    div_term = integ((dux_dx + duy_dy) * phi)

    # weak residuals
    R_x   = grad_term_x + p_term_x - f_term_x
    R_y   = grad_term_y + p_term_y - f_term_y
    R_div = div_term

    u2_patch = u_x_patch ** 2 + u_y_patch ** 2  # [B,N,Ky,Kx]
    grad2_patch = dux_dx ** 2 + dux_dy ** 2 + duy_dx ** 2 + duy_dy ** 2
    grad_energy_patch = integ(nu_patch * grad2_patch)  # [B,N]
    u_energy_patch = integ(u2_patch)  # [B,N]

    lam = 0.0
    energy_patch = grad_energy_patch + lam * u_energy_patch
    energy_norm_patch = torch.sqrt(energy_patch + 1e-8)

    R_x_norm = R_x / (energy_norm_patch + 1e-8)
    R_y_norm = R_y / (energy_norm_patch + 1e-8)
    R_div_norm = R_div / (energy_norm_patch + 1e-8)

    residual = torch.stack([R_x_norm, R_y_norm, R_div_norm], dim=1)

    return residual, center_coords



class WeakStokesResidual:
    def __init__(self,
                 data,
                 x_range=(0., 1.),
                 y_range=(0., 1.),
                 sigma_range=(3., 10.),
                 test_fun='wd_wv',
                 lam_bc=0.0,
                 F0=0.0,
                 ):
        """
        Wrapper for weak Stokes residual and optional boundary residual utility.

        data:
            Dataset object with .denormalize_data(u) and .denormalize_alpha(nu).
        """
        self.data = data
        self.ranges = [x_range, y_range]
        self.sigma_range = sigma_range

        self.test_fun = test_fun
        self.lam_bc = lam_bc

        self.res_scaling = 1e4
        self.bc_scaling = 1.0
        self.f0 = F0


    def compute_residual(self, u, nu, denormalize=True, pretrain=False):
        """
        Compute the weak-form residual.

        Args:
            u  : [B, 3, H, H] - solution fields [u_x, u_y, p]
            nu : [B, H, H] or [B,1,H,H] - viscosity field

        Returns:
            torch.Tensor with shape ``(B,)``.
        """
        if denormalize:
            u = self.data.denormalize_data(u)
            nu = self.data.denormalize_alpha(nu)

        residual, _ = compute_weak_stokes_residual(
            u, nu,
            self.ranges,
            sigma_range=self.sigma_range,
            test_fun=self.test_fun,
            F0=self.f0,
        )

        # residual: [B,3,N]; square, sum over equations, mean over tests
        residual = residual.pow(2).sum(1).mean(-1) * self.res_scaling

        return residual


    def compute_boundary_constraint(
            self,
            u: torch.Tensor,
            U_lid: float = 1.0,
    ) -> torch.Tensor:
        """
        Boundary-condition residual for the Stokes lid-driven cavity:

            u = (0, 0)                        on y=0, x=0, x=1
            u = (U_lid * sin^2(pi x), 0)      on y=1.

        Parameters
        ----------
        u : (B, 3, H, W)
            Velocity-pressure field with channels [u_x, u_y, p].
        U_lid : float
            Lid velocity scale.

        Returns
        -------
        R_bc : (B,)
            Boundary integral of squared Dirichlet mismatch over all sides.
        """
        assert u.ndim == 4 and u.size(1) == 3, "u must be [B,3,H,W]"
        B, _, H, W = u.shape
        device = u.device
        dtype = u.dtype

        x_range, y_range = self.ranges
        dx = (x_range[1] - x_range[0]) / (W - 1)
        dy = (y_range[1] - y_range[0]) / (H - 1)

        u_x = u[:, 0]  # (B,H,W)
        u_y = u[:, 1]

        # ------------------------------------------------------------------
        # Bottom boundary: y = 0, no-slip (0,0)
        # ------------------------------------------------------------------
        u_x_bot = u_x[:, 0, :]  # (B,W)
        u_y_bot = u_y[:, 0, :]

        r_x_bot = u_x_bot  # target is 0
        r_y_bot = u_y_bot
        mag2_bot = r_x_bot ** 2 + r_y_bot ** 2  # (B,W)

        R_bot = mag2_bot.sum(dim=1) * dx  # integrate along x

        # ------------------------------------------------------------------
        # Top boundary: y = 1, moving lid
        #   u_x = U_lid * sin^2(pi x), u_y = 0
        # ------------------------------------------------------------------
        x0, x1 = x_range
        xs = torch.linspace(x0, x1, W, device=device, dtype=dtype)  # (W,)
        lid_profile = U_lid * torch.sin(math.pi * xs) ** 2  # (W,)

        u_x_top = u_x[:, -1, :]  # (B,W)
        u_y_top = u_y[:, -1, :]

        r_x_top = u_x_top - lid_profile.unsqueeze(0)  # broadcast over batch
        r_y_top = u_y_top  # target is 0
        mag2_top = r_x_top ** 2 + r_y_top ** 2  # (B,W)

        R_top = mag2_top.sum(dim=1) * dx

        # ------------------------------------------------------------------
        # Left boundary: x = 0, no-slip (0,0)
        # ------------------------------------------------------------------
        u_x_left = u_x[:, :, 0]  # (B,H)
        u_y_left = u_y[:, :, 0]

        r_x_left = u_x_left
        r_y_left = u_y_left
        mag2_left = r_x_left ** 2 + r_y_left ** 2  # (B,H)

        R_left = mag2_left.sum(dim=1) * dy  # integrate along y

        # ------------------------------------------------------------------
        # Right boundary: x = 1, no-slip (0,0)
        # ------------------------------------------------------------------
        u_x_right = u_x[:, :, -1]  # (B,H)
        u_y_right = u_y[:, :, -1]

        r_x_right = u_x_right
        r_y_right = u_y_right
        mag2_right = r_x_right ** 2 + r_y_right ** 2  # (B,H)

        R_right = mag2_right.sum(dim=1) * dy  # integrate along y

        # total Dirichlet BC residual over all four sides
        R_bc = R_bot + R_top + R_left + R_right  # (B,)

        return R_bc * self.bc_scaling
