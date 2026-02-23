"""
Plane-strain elasticity residuals on 2-D Cartesian grids.

Mathematical specification
--------------------------
Target PDE (no body force):
``-div sigma(u;E) = 0``, with
``sigma = 2 mu eps + lambda tr(eps) I``,
``mu = E/(2(1+nu))``,
``lambda = nu E / ((1+nu)(1-2nu))``.

Implemented residuals:
1. Strong residual (`compute_strong_elasticity_residual`):
   computes ``r = -div sigma`` from finite-difference derivatives and returns
   ``int_Omega (r_x^2 + r_y^2) dx``.
2. Weak residual (`compute_weak_elasticity_residual`):
   uses scalar test family ``psi_n`` and forms per-component moments
   ``R_x,n = int sigma_x:grad(psi_n e_x) dx``,
   ``R_y,n = int sigma_y:grad(psi_n e_y) dx``.
   The implementation computes these via patch quadrature from
   `prepare_test_functions_nd` and normalizes by patch mean ``E``.

Inputs/Outputs
--------------
- Displacement channel order is always ``[u_x, u_y]``.
- Weak residual function returns shape ``(B,2,N_test)`` and center coordinates.
- Wrapper `WeakElasticityResidual.compute_residual` returns shape ``(B,)`` after
  squaring, summing equation channels, averaging tests, and applying scaling.
"""

import torch

from residuals.common import prepare_test_functions_nd

def compute_strong_elasticity_residual(u: torch.Tensor,
                                     E: torch.Tensor,
                                     ranges=None,            # [(x0,x1), (y0,y1)]
                                     nu: float = 0.30):
    """
    Strong-form residual for 2-D plane-strain linear elasticity (isotropic).
    Residual vector field r = -div σ(u;E), with
        σ = 2 μ(E) ε(u) + λ(E) tr(ε(u)) I,
        μ(E) = E / (2(1+ν)),  λ(E) = ν E / ((1+ν)(1-2ν)).

    Parameters
    ----------
    u      : (B, 2, H, W)  displacement field [u_x, u_y]
    E      : (B, 1, H, W)  Young's modulus field
    ranges : [(x0,x1), (y0,y1)]  physical extents in x (width) and y (height)
    nu     : Poisson's ratio in (0, 0.5)

    Returns
    -------
    ‖r‖_{L2}^2 : (B,)  squared L2 norm of the strong residual over the domain
    """
    if ranges is None:
        ranges = [(0., 1.), (0., 1.)]

    assert u.ndim == 4 and u.size(1) == 2, "u must be (B,2,H,W)"
    assert E.ndim == 4 and E.size(1) == 1, "E must be (B,1,H,W)"
    B, _, H, W = u.shape

    # grid spacings (match ordering used in torch.gradient below: dims (2,3) -> (y,x))
    dy, dx = [(r[1] - r[0]) / (n - 1) for r, n in zip(ranges, (H, W))]
    dA = dx * dy

    # material params (broadcast to (B,H,W))
    E_ = E.squeeze(1)
    mu  = E_ / (2.0 * (1.0 + nu))
    lam = (nu * E_) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # displacements
    ux = u[:, 0]  # (B,H,W)
    uy = u[:, 1]

    # first derivatives (central, edge_order=2), dims: y=2, x=3
    dux_dy, dux_dx = torch.gradient(ux, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    duy_dy, duy_dx = torch.gradient(uy, spacing=(dy, dx), dim=(1, 2), edge_order=2)

    # small-strain tensor components
    exx = dux_dx
    eyy = duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    tr  = exx + eyy

    # Cauchy stress components
    sxx = 2.0 * mu * exx + lam * tr
    syy = 2.0 * mu * eyy + lam * tr
    sxy = 2.0 * mu * exy  # = syx

    # stress divergences
    dsxx_dy, dsxx_dx = torch.gradient(sxx, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dsxy_dy, dsxy_dx = torch.gradient(sxy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dsyy_dy, dsyy_dx = torch.gradient(syy, spacing=(dy, dx), dim=(1, 2), edge_order=2)

    div_sigma_x = dsxx_dx + dsxy_dy
    div_sigma_y = dsxy_dx + dsyy_dy

    # strong-form residual r = -div σ  (no body force here)
    rx = -div_sigma_x
    ry = -div_sigma_y

    # L2 norm over domain (sum of components)
    R_L2_sq = ((rx**2 + ry**2).sum(dim=(1, 2))) * dA  # (B,)

    return R_L2_sq


class WeakElasticityResidual:
    """Training/evaluation wrapper around weak elasticity residual moments."""
    def __init__(self,
                 data,
                 x_range=(0., 1.),
                 y_range=(0., 1.),
                 sigma_range=(3., 10.),
                 test_fun='wd_wv',
                 lam_bc=0.0
                 ):

        self.data = data
        self.ranges = [x_range, y_range]
        self.sigma_range = sigma_range

        self.test_fun = test_fun
        self.lam_bc = lam_bc

        self.res_scaling = 1e7
        self.bc_scaling = 1e4  # maximum value is 0.1

    def compute_residual(self, u, E, denormalize=True, pretrain=False):
        """
        Compute scaled weak residual and optional boundary mismatch penalty.

        Args:
            u: ``(B,2,H,W)`` displacement.
            E: ``(B,1,H,W)`` or ``(B,H,W)`` Young's modulus.

        Returns:
            torch.Tensor of shape ``(B,)``.
        """
        if denormalize:
            u = self.data.denormalize_data(u)
            E = self.data.denormalize_alpha(E)

        residual, _  = compute_weak_elasticity_residual(u, E,
                                                  self.ranges,
                                                  sigma_range=self.sigma_range,
                                                  test_fun=self.test_fun)

        residual = residual.pow(2).sum(1).mean(-1) * self.res_scaling

        if pretrain:
            return residual
        else:
            res_bc = self.compute_boundary_constraint(u).mean() * self.lam_bc
            return residual + res_bc


    def compute_residual_map(self, u, E, denormalize=True, test_fun='wd', sigma=10.):
        """
        Compute per-test weak residual magnitude map reshaped to ``(B,H,W)``.

        Args:
            u: ``(B,2,H,W)`` displacement.
            E: ``(B,1,H,W)`` or ``(B,H,W)`` Young's modulus.

        Returns:
            torch.Tensor with shape ``(B,H,W)``.
        """
        if denormalize:
            u = self.data.denormalize_data(u)
            E = self.data.denormalize_alpha(E)

        residual, _  = compute_weak_elasticity_residual(u, E,
                                                  self.ranges,
                                                  sigma_range=[sigma, sigma],
                                                  test_fun=test_fun)

        B, C, H, W = u.shape
        residual_map = residual.pow(2).sum(1).pow(0.5)


        return residual_map.view(B, H, W)


    def compute_boundary_constraint(self, u, A_bc_bot=0.075, A_bc_top=0.1):
        """
        Compute top/bottom Dirichlet mismatch penalty used by this wrapper.

        Args:
            u: ``(B,2,H,W)`` displacement.
            A_bc_bot: bottom sine amplitude for ``u_y`` target.
            A_bc_top: top sine amplitude for ``u_y`` target.

        Returns:
            torch.Tensor of shape ``(B,)``.
        """
        if u.ndim == 3:  # [B,C,H,W]
            u = u.unsqueeze(1)


        x = torch.linspace(0.0, 1.0, u.shape[-1], device=u.device, dtype=torch.float32)
        u_top_y = A_bc_top * torch.sin(torch.pi * x)
        diff_top_y = (u[:, 1, 0, :] - u_top_y).pow(2).mean(dim=-1)  # [B,C,W]
        diff_top_x = (u[:, 0, 0, :] - 0.0).pow(2).mean(dim=-1)
        diff_top = (diff_top_x + diff_top_y) * 0.5

        u_bot_y = -A_bc_bot * torch.sin(torch.pi * x)
        diff_bot_y = (u[:, 1, -1, :] - u_bot_y).pow(2).mean(dim=-1)  # [B,C,W]
        diff_bot_x = (u[:, 0, -1, :] - 0.0).pow(2).mean(dim=-1)
        diff_bot = (diff_bot_x + diff_bot_y) * 0.5

        diff2 = 0.5 * (diff_top + diff_bot)
        return diff2 * self.bc_scaling



def compute_weak_elasticity_residual(u: torch.Tensor,
                                     E: torch.Tensor,
                                     ranges,                 # [(x0,x1),(y0,y1)]
                                     nu: float = 0.30,
                                     sigma_range=(5., 20.),
                                     test_fun: str = 'wd_wv',
                                     N_test=None):
    """
    Compute weak residual moments for 2-D plane-strain isotropic elasticity.

    Discrete estimator:
    ``R_x,n = int (sigma_xx d_x psi_n + sigma_xy d_y psi_n) dx``,
    ``R_y,n = int (sigma_xy d_x psi_n + sigma_yy d_y psi_n) dx``,
    where integrals are patchwise trapezoidal sums from
    ``prepare_test_functions_nd``.

    Inputs
    ------
    u : [B, 2, H, W]           displacement field (u_x, u_y)
    E : [B, 1, H, W] or [B,H,W] Young's modulus
    ranges : [(x0,x1), (y0,y1)] physical extents
    nu : Poisson's ratio
    sigma_range, test_fun, N_test : passed through to your test-function generator

    Returns
    -------
    residual : [B, 2, N_test]  weak residual per component and test
    centers  : list of center coordinates per axis.
    """
    assert u.ndim == 4 and u.size(1) == 2, "u must be [B,2,H,W]"
    B, _, H, W = u.shape
    device = u.device

    # Accept E as [B,1,H,W] or [B,H,W]
    if E.ndim == 4 and E.size(1) == 1:
        E_s = E[:, 0]                        # [B,H,W]
    elif E.ndim == 3:
        E_s = E
    else:
        raise ValueError("E must be [B,1,H,W] or [B,H,W].")

    # Fold channels into batch so both displacement components use identical test
    # patches and quadrature nodes.
    u_bc = u.reshape(B * 2, H, W)            # [BC,H,W], BC=B*2
    E_bc = E_s.repeat_interleave(2, dim=0)   # [BC,H,W]

    (u_patch, E_patch, _, grad_u_patch, phi_moll,
     grad_phi_moll, center_coords, weights_patch, dA) = prepare_test_functions_nd(
        u_bc, E_bc, ranges,
        device=device,
        sigma_range=sigma_range,
        test_fun=test_fun,
        N_test=N_test
    )
    # Shapes:
    #   grad_u_patch : [BC, N, Ky, Kx, 2]  (last dim = (∂/∂y, ∂/∂x))
    #   E_patch      : [BC, N, Ky, Kx]
    #   grad_phi_moll: [N,  Ky, Kx, 2]     (components (∂/∂y, ∂/∂x))
    #   weights_patch: [N,  Ky, Kx]
    #   dA           : scalar

    N = grad_u_patch.shape[1]
    Ky, Kx = grad_u_patch.shape[2], grad_u_patch.shape[3]

    # Unfold component dimension back out: [B, 2, N, Ky, Kx, 2]
    grad_u_patch = grad_u_patch.view(B, 2, N, Ky, Kx, 2)
    # E was duplicated across the folded components; keep one copy: [B, N, Ky, Kx]
    E_patch = E_patch.view(B, 2, N, Ky, Kx)[:, 0]

    # Gradient components (remember ordering in last dim)
    dux_dy = grad_u_patch[:, 0, ..., 0]   # ∂y u_x
    dux_dx = grad_u_patch[:, 0, ..., 1]   # ∂x u_x
    duy_dy = grad_u_patch[:, 1, ..., 0]   # ∂y u_y
    duy_dx = grad_u_patch[:, 1, ..., 1]   # ∂x u_y

    # Small-strain
    exx = dux_dx
    eyy = duy_dy
    exy = 0.5 * (dux_dy + duy_dx)

    # Lamé parameters on the patch
    mu  = E_patch / (2.0 * (1.0 + nu))
    lam = (nu * E_patch) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    tr  = exx + eyy
    sxx = 2.0 * mu * exx + lam * tr
    syy = 2.0 * mu * eyy + lam * tr
    sxy = 2.0 * mu * exy

    # Contract σ with ∇phi (grad_phi_moll[...,0]=∂yφ, grad_phi_moll[...,1]=∂xφ)
    gpy = grad_phi_moll[..., 0]            # [N,Ky,Kx]
    gpx = grad_phi_moll[..., 1]            # [N,Ky,Kx]
    W   = weights_patch                    # [N,Ky,Kx]

    # Integrate over each patch: sum_{Ky,Kx} ( · ) * W * dA  -> [B,N]
    def integ(t):
        return (t * W * dA).sum(dim=(-2, -1))

    lhs_x = integ(sxx * gpx + sxy * gpy)   # (σ∇φ)_x, [B,N]
    lhs_y = integ(sxy * gpx + syy * gpy)   # (σ∇φ)_y, [B,N]

    # No body force ⇒ RHS = 0
    residual = torch.stack([lhs_x, lhs_y], dim=1)   # [B, 2, N]

    mean_E = (E_patch * weights_patch).sum(dim=(-1, -2)) / weights_patch.sum((-1, -2))
    residual = residual / (mean_E.unsqueeze(1) + 1e-8)

    return residual, center_coords
