"""
Complex Helmholtz residuals (real/imag split) on 2-D grids.

Mathematical specification
--------------------------
Target PDE:
``(-Delta - (1 - i*tan_delta) kappa^2) u = s``,
``kappa^2 = omega^2 / c^2``,
with ``u = u_R + i u_I`` and fixed real source ``s`` from `fixed_source`.

Implemented residuals:
1. Strong residual (`compute_strong_helmholtz_residual`):
   pointwise
   ``r_R = -Delta u_R - kappa^2 u_R + alpha_I u_I - s``,
   ``r_I = -Delta u_I - kappa^2 u_I - alpha_I u_R``,
   ``alpha_I = -tan_delta * kappa^2``.
   Returns relative squared residual
   ``int (r_R^2+r_I^2) / int (|grad u|^2 + kappa^2 |u|^2)``.
2. Weak residual (`compute_weak_helmholtz_residual`):
   for test family ``psi_n``,
   ``R_R,n = int (grad u_R·grad psi_n - kappa^2 u_R psi_n + alpha_I u_I psi_n - s psi_n)``,
   ``R_I,n = int (grad u_I·grad psi_n - kappa^2 u_I psi_n - alpha_I u_R psi_n)``.
   Patch integrals are from `prepare_test_functions_nd` and normalized by
   ``sqrt(int (|grad u|^2 + kappa^2|u|^2))`` per patch/test.

Inputs/Outputs
--------------
- Complex field convention is channel order ``[real, imag]`` with shapes
  ``(B,2,H,W)``.
- `compute_weak_helmholtz_residual` returns ``(B,2,N_test)`` plus centers.
- Wrapper `WeakHelmholtzResidual.compute_residual` returns shape ``(B,)`` after
  squared aggregation and scaling.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from residuals.common import prepare_test_functions_nd

def fixed_source(N: int, sigma: float = 0.05, amplitude: float = 1.0) -> torch.tensor:
    """Return centered Gaussian real source on an ``N x N`` grid."""
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, N),
        np.linspace(0.0, 1.0, N),
        indexing="ij"
    )
    xc, yc = 0.5, 0.5
    r2 = (x - xc)**2 + (y - yc)**2
    return torch.tensor(amplitude * np.exp(-r2 / (2.0 * sigma**2)), dtype=torch.float32)



def compute_strong_helmholtz_residual(
    u: torch.Tensor,
    c: torch.Tensor,
    ranges=None,              # [(x0,x1), (y0,y1)]
    omega: float = 20.0,
    loss_tan: float = 0.02,
):
    """
    Compute relative strong-form Helmholtz residual for each batch sample.

        (-Δ - (1 - i*loss_tan) * kappa^2(x)) u(x) = s(x),
        kappa^2(x) = omega^2 / c(x)^2.

    We split u = u_R + i u_I and compute the real residuals

        r_R = -Δ u_R - kappa^2 u_R + alpha_I u_I - s_R,
        r_I = -Δ u_I - kappa^2 u_I - alpha_I u_R - s_I,

    where alpha_I = -loss_tan * kappa^2.

    Parameters
    ----------
    u : (B, 2, H, W)
        Complex-valued field as two channels [real, imag].
    c : (B, 1, H, W) or (B, H, W)
        Sound speed field c(x).
    ranges : [(x0,x1), (y0,y1)]
        Physical extents in x and y.
    omega : float
        Angular frequency.
    loss_tan : float
        Loss tangent (>=0). 0 corresponds to lossless Helmholtz.
    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` with relative squared strong residual.
    """
    if ranges is None:
        ranges = [(0., 1.), (0., 1.)]

    assert u.ndim == 4 and u.size(1) == 2, "u must be (B,2,H,W)"
    B, _, H, W = u.shape
    device = u.device

    # grid spacings
    dy, dx = [(r[1] - r[0]) / (n - 1) for r, n in zip(ranges, (H, W))]
    dA = dx * dy

    # sound speed field to shape (B,H,W)
    if c.ndim == 4 and c.size(1) == 1:
        c_s = c[:, 0]
    elif c.ndim == 3:
        c_s = c
    else:
        raise ValueError("c must be [B,1,H,W] or [B,H,W].")

    # wave number squared and complex splitting
    kappa2 = (omega**2) / (c_s**2)                   # (B,H,W), real
    alpha_R = kappa2                                 # real part
    alpha_I = -loss_tan * kappa2                    # imag part

    # source (real/imag)

    s_R = fixed_source(H).to(device)
    s_I = torch.zeros_like(s_R, device=device)

    # real/imag parts of u
    u_R = u[:, 0]   # (B,H,W)
    u_I = u[:, 1]

    # first derivatives
    duR_dy, duR_dx = torch.gradient(u_R, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    duI_dy, duI_dx = torch.gradient(u_I, spacing=(dy, dx), dim=(1, 2), edge_order=2)

    # second derivatives -> Laplacian
    d2uR_dy2, _ = torch.gradient(duR_dy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    _, d2uR_dx2 = torch.gradient(duR_dx, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    lap_u_R = d2uR_dy2 + d2uR_dx2

    d2uI_dy2, _ = torch.gradient(duI_dy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    _, d2uI_dx2 = torch.gradient(duI_dx, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    lap_u_I = d2uI_dy2 + d2uI_dx2

    # strong residual components
    r_R = -lap_u_R - alpha_R * u_R + alpha_I * u_I - s_R
    r_I = -lap_u_I - alpha_R * u_I - alpha_I * u_R - s_I

    # squared L2 norm over domain (sum over real+imag)
    res_sq = ((r_R ** 2 + r_I ** 2).sum(dim=(1, 2))) * dA  # (B,)

    # --- denominator: global energy matching weak-form energy_patch ---
    u2 = u_R ** 2 + u_I ** 2
    grad2 = duR_dx ** 2 + duR_dy ** 2 + duI_dx ** 2 + duI_dy ** 2

    energy = ((grad2 + kappa2 * u2).sum(dim=(1, 2))) * dA  # (B,)

    R_rel_sq = res_sq / (energy + 1e-8)  # dimensionless, squared

    return R_rel_sq

def compute_weak_helmholtz_residual(
    u: torch.Tensor,
    c: torch.Tensor,
    ranges,                       # [(x0,x1),(y0,y1)]
    omega: float,
    loss_tan: float = 0.02,
    sigma_range=(5.0, 20.0),
    test_fun: str = "wd_wv",
    N_test=None,
):
    """
    Compute weak residual moments for real and imaginary Helmholtz equations.

        (-Δ - (1 - i*loss_tan) * kappa^2(x)) u = s,
        kappa^2(x) = omega^2 / c(x)^2.

    We split u = u_R + i u_I and test the real and imaginary equations with the
    same scalar test functions φ_n:

        R_R(u; φ_n) = ∫ (∇u_R·∇φ_n - α_R u_R φ_n + α_I u_I φ_n) dx,
        R_I(u; φ_n) = ∫ (∇u_I·∇φ_n - α_R u_I φ_n - α_I u_R φ_n) dx,

    where α_R = kappa^2, α_I = -loss_tan * kappa^2.

    The implementation includes the fixed source term from ``fixed_source(H)``
    in the real equation only.

    Parameters
    ----------
    u : [B, 2, H, W]
        Complex field as two channels [real, imag].
    c : [B, 1, H, W] or [B, H, W]
        Sound speed field.
    ranges : [(x0,x1),(y0,y1)]
        Physical extents.
    omega : float
        Angular frequency.
    loss_tan : float
        Loss tangent.
    sigma_range, test_fun, N_test :
        Passed through to prepare_test_functions_nd.

    Returns
    -------
    residual : [B, 2, N_test]
        Weak residual per channel (real/imag) and test function.
    centers  : center_coords from prepare_test_functions_nd.
    """
    assert u.ndim == 4 and u.size(1) == 2, "u must be [B,2,H,W]"
    B, _, H, W = u.shape
    device = u.device

    # sound speed to [B,H,W]
    if c.ndim == 4 and c.size(1) == 1:
        c_s = c[:, 0]
    elif c.ndim == 3:
        c_s = c
    else:
        raise ValueError("c must be [B,1,H,W] or [B,H,W].")

    # fold channels into batch to reuse scalar TF pipeline
    u_bc = u.reshape(B * 2, H, W)              # [B*2,H,W]
    c_bc = c_s.repeat_interleave(2, dim=0)     # [B*2,H,W]

    s = fixed_source(H).to(device)

    (u_patch, c_patch, s_patch, grad_u_patch, phi_moll,
     grad_phi_moll, center_coords, weights_patch, dA) = prepare_test_functions_nd(
        u_bc, c_bc, ranges,
        source=s,
        device=device,
        sigma_range=sigma_range,
        test_fun=test_fun,
        N_test=N_test,
    )
    # Shapes (as in elasticity):
    #   u_patch        : [BC, N, Ky, Kx]
    #   c_patch        : [BC, N, Ky, Kx]
    #   s_patch        : [N, Ky, Kx]
    #   grad_u_patch   : [BC, N, Ky, Kx, 2]   (∂/∂y, ∂/∂x)
    #   phi_moll       : [N,  Ky, Kx]
    #   grad_phi_moll  : [N,  Ky, Kx, 2]
    #   weights_patch  : [N,  Ky, Kx]
    #   dA             : scalar

    BC = u_bc.shape[0]
    N = grad_u_patch.shape[1]
    Ky, Kx = grad_u_patch.shape[2], grad_u_patch.shape[3]

    # unfold component dimension: [B, 2, N, Ky, Kx, 2]
    grad_u_patch = grad_u_patch.view(B, 2, N, Ky, Kx, 2)
    u_patch = u_patch.view(B, 2, N, Ky, Kx)
    c_patch = c_patch.view(B, 2, N, Ky, Kx)[:, 0]  # [B,N,Ky,Kx], same c for both channels

    # gradient components
    duR_dy = grad_u_patch[:, 0, ..., 0]  # [B,N,Ky,Kx]
    duR_dx = grad_u_patch[:, 0, ..., 1]
    duI_dy = grad_u_patch[:, 1, ..., 0]
    duI_dx = grad_u_patch[:, 1, ..., 1]

    # u on patches
    u_R_patch = u_patch[:, 0]  # [B,N,Ky,Kx]
    u_I_patch = u_patch[:, 1]

    # kappa^2 and alpha on patches
    kappa2_patch = (omega ** 2) / (c_patch ** 2  + 1e-8)
    alpha_R_patch = kappa2_patch
    alpha_I_patch = -loss_tan * kappa2_patch

    # test function data
    gpy = grad_phi_moll[..., 0]  # [N,Ky,Kx], ∂y φ
    gpx = grad_phi_moll[..., 1]  # [N,Ky,Kx], ∂x φ
    phi = phi_moll  # [N,Ky,Kx]
    W = weights_patch  # [N,Ky,Kx]

    # helper: integrate over each patch -> [B,N]
    def integ(t: torch.Tensor) -> torch.Tensor:
        # t is [B,N,Ky,Kx]
        return (t * W * dA).sum(dim=(-2, -1))

    # gradient terms ∫ ∇u·∇φ
    grad_term_R = integ(duR_dy * gpy + duR_dx * gpx)  # [B,N]
    grad_term_I = integ(duI_dy * gpy + duI_dx * gpx)

    # mass/coupling terms
    mass_R_uR = integ(alpha_R_patch * u_R_patch * phi)
    mass_R_uI = integ(alpha_R_patch * u_I_patch * phi)
    mass_I_uR = integ(alpha_I_patch * u_R_patch * phi)
    mass_I_uI = integ(alpha_I_patch * u_I_patch * phi)

    # source term: ∫ s φ dx, shared across batch
    # s_patch: [N,Ky,Kx], φ: [N,Ky,Kx], W: [N,Ky,Kx]
    src_per_test = (s_patch * phi * W * dA).sum(dim=(-2, -1))  # [N]
    # broadcast to [B,N]
    src_term = src_per_test.unsqueeze(0).expand(B, -1)


    # energy normalisation
    u2_patch = u_R_patch ** 2 + u_I_patch ** 2  # [B,N,Ky,Kx]
    grad2_patch = duR_dx ** 2 + duR_dy ** 2 + duI_dx ** 2 + duI_dy ** 2
    energy_patch = integ(grad2_patch + kappa2_patch * u2_patch)  # [B,N]

    # weak residuals
    # R_R = ∫(∇u_R·∇φ - α_R u_R φ + α_I u_I φ - s φ) dx
    # R_I = ∫(∇u_I·∇φ - α_R u_I φ - α_I u_R φ          ) dx
    R_R = grad_term_R - mass_R_uR + mass_I_uI - src_term
    R_I = grad_term_I - mass_R_uI - mass_I_uR

    eps = 1e-8
    R_R_norm = R_R / (torch.sqrt(energy_patch) + eps)
    R_I_norm = R_I / (torch.sqrt(energy_patch) + eps)

    residual = torch.stack([R_R_norm, R_I_norm], dim=1)  # [B,2,N]


    return residual, center_coords



class WeakHelmholtzResidual:
    """Wrapper that converts weak Helmholtz moments into a training scalar."""
    def __init__(self,
                 data,
                 x_range=(0., 1.),
                 y_range=(0., 1.),
                 sigma_range=(3., 10.),
                 test_fun='wd_wv',
                 lam_bc=0.0,
                 omega=20.,
                 loss_tan=0.02,
                 ):

        self.data = data
        self.ranges = [x_range, y_range]
        self.sigma_range = sigma_range

        self.test_fun = test_fun
        self.lam_bc = lam_bc

        self.res_scaling = 1e5
        self.bc_scaling = 1.0


        self.omega = omega
        self.loss_tan = loss_tan


    def compute_residual(self, u, c, denormalize=True, pretrain=False):
        """
        Compute scaled weak residual.

        Args:
            u: ``(B,2,H,W)`` complex field split into real/imag channels.
            c: ``(B,1,H,W)`` or ``(B,H,W)`` sound speed.

        Returns:
            torch.Tensor of shape ``(B,)``.
        """
        if denormalize:
            u = self.data.denormalize_data(u)
            c = self.data.denormalize_alpha(c)

        residual, _  = compute_weak_helmholtz_residual(u, c,
                                                       self.ranges,
                                                       omega=self.omega,
                                                       loss_tan=self.loss_tan,
                                                       sigma_range=self.sigma_range,
                                                       test_fun=self.test_fun)

        residual = residual.pow(2).sum(1).mean(-1) * self.res_scaling

        return residual

    def compute_boundary_constraint(self, u: torch.Tensor) -> torch.Tensor:
        """
        Return zero boundary penalty placeholder (boundary term not used here).
        """
        B, _, H, W = u.shape
        return torch.zeros(B, 1)

