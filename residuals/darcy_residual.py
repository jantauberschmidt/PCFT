"""
Darcy residuals on 2-D Cartesian grids.

Mathematical specification
--------------------------
Target PDE:
``-div(a grad u) = f`` on ``Omega=[x0,x1]x[y0,y1]``.

Implemented residuals:
1. Strong residual (`compute_strong_darcy_residual`):
   ``r = -div(a grad u) - f`` computed pointwise with `torch.gradient`
   derivatives (central interior, one-sided boundaries, `edge_order=2`).
   Returns ``int_Omega r^2 dx`` per batch sample.
2. Weak residual (`compute_weak_darcy_residual`):
   for test family ``psi_n`` from `residuals.common.prepare_test_functions_nd`,
   computes
   ``R_n = int_Omega a grad u · grad psi_n dx - int_Omega f psi_n dx``.
   This corresponds to integration-by-parts weak form with source term and no
   explicit boundary term (test mollifier vanishes at boundary).
   Residuals are normalized by local mean permeability on each patch.

Inputs/Outputs
--------------
- ``u`` and ``a`` are real-valued tensors on the same device.
- Batch semantics:
  - strong: ``u`` accepted as ``(B,1,H,W)`` and squeezed to ``(B,H,W)``.
  - weak: ``u,a`` expected as ``(B,H,W)`` (wrappers also accept channel dim).
- `WeakDarcyResidual.compute_residual` returns per-sample penalized residual
  (shape ``(B,)``), scaled by `res_scaling`, optionally plus boundary penalty.
"""

import math
import torch
import torch.nn.functional as F

from residuals.common import prepare_test_functions_nd


def compute_strong_darcy_residual(u: torch.Tensor,
                                  a: torch.Tensor,
                                  ranges=None,           # [(x0,x1),(y0,y1)]
                                  f: float = 1.0):
    """
    Compute strong-form Darcy residual energy ``int_Omega r^2 dx``.

    Parameters
    ----------
    u      : (B,1,H,W) pressure field
    a      : (B,H,W) permeability field
    ranges : [(x0,x1), (y0,y1)]
    f      : scalar RHS (constant)

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` with squared ``L^2`` norm of ``r=-div(a grad u)-f``.
    """

    if ranges is None:
        ranges = [(0., 1.), (0., 1.)]

    B, _, H, W = u.shape
    u = u.view(B, H, W)
    dx, dz  = [(r[1] - r[0]) / (n - 1) for r, n in zip(ranges, (H, W))]
    dA      = dx * dz                    # cell area

    # --- grad u -------------------------------------------------------
    du_dx, du_dz = torch.gradient(u, spacing=(dx, dz), dim=(1, 2), edge_order=2)

    # --- flux q = κ ∇u -----------------------------------------------
    qx = a * du_dx
    qz = a * du_dz

    # --- divergence  ∂x qx + ∂z qz ------------------------------------
    dqx_dx, _ = torch.gradient(qx, spacing=(dx, dz), dim=(1, 2), edge_order=2)
    _, dqz_dz = torch.gradient(qz, spacing=(dx, dz), dim=(1, 2), edge_order=2)
    div_q     = dqx_dx + dqz_dz

    # --- strong residual  R = -div(q) - f -----------------------------
    R = -div_q - f

    # --- L2 norm over Ω ----------------------------------------------
    R_L2_sq = (R.pow(2).sum(dim=(1, 2))) * dA  # shape (B,)

    return R_L2_sq


def compute_weak_darcy_residual(u, a, ranges, f=1.0, sigma_range=(5., 20.), test_fun='wd_wv', N_test=None):
    """Compute patchwise weak residuals for Darcy equation.

    Returns
    -------
    residual : torch.Tensor
        Shape ``(B, N_test)`` with normalized weak residual values.
    center_coords : list[torch.Tensor]
        Test-function centers returned by ``prepare_test_functions_nd``.
    """
    B, *axes = u.shape
    device = u.device


    (u_patch, a_patch, _, grad_u_patch, phi_moll,
     grad_phi_moll, center_coords, weights_patch, dA) = prepare_test_functions_nd(u, a, ranges,
                                                                                  device=device,
                                                                                  sigma_range=sigma_range,
                                                                                  test_fun=test_fun,
                                                                                  N_test=N_test)


    integrand = a_patch * (grad_u_patch * grad_phi_moll).sum(-1)  # [B, Kx, Ky]
    lhs = (integrand * weights_patch * dA).sum(dim=(-1, -2))  # [B]

    rhs = (f * phi_moll * weights_patch * dA).sum(dim=(-1, -2))  # scalar
    residual = (lhs - rhs)

    # # Normalize residual by permeability since errors get amplified by a factor of a
    mean_a = (a_patch * weights_patch).sum(dim=(-1, -2)) / weights_patch.sum((-1, -2))
    residual = residual / (mean_a + 1e-8)

    if N_test is None:
        N_test = math.prod(axes)

    return residual.view(B, N_test), center_coords


class WeakDarcyResidual:
    """
    Wrapper that builds a training residual from weak Darcy test integrals.

    Final per-sample objective:
    ``res_scaling * mean_n(R_n^2) + lam_bc * mean_b((u-u_bc)^2) * bc_scaling``.
    """

    def __init__(self,
                 data,
                 f=1.0,
                 x_range = (0., 1.),
                 y_range = (0., 1.),
                 sigma_range=(2., 20.),
                 test_fun='wd_wv',
                 lam_bc=0.0
                 ):
        """Initialize weak Darcy residual hyperparameters and ranges."""
        self.data = data

        self.f = f
        self.ranges = [x_range, y_range]
        self.sigma_range = sigma_range
        self.test_fun = test_fun
        self.lam_bc = lam_bc

        self.res_scaling = 1e8
        self.bc_scaling = 1e6


    def compute_residual(self, u, a, denormalize=True, pretrain=False):
        """
        Compute the scaled weak residual with optional boundary penalty.

        Args:
            u: ``(B,1,H,W)`` or ``(B,H,W)`` predicted solution.
            a: ``(B,1,H,W)`` or ``(B,H,W)`` permeability.

        Returns:
            torch.Tensor of shape ``(B,)``.
        """
        if denormalize:
            u = self.data.denormalize_data(u)
            a = self.data.denormalize_alpha(a)

        # remove channel dimension for residual computation
        if u.ndim == 4:
            u = u.squeeze(1)
        if a.ndim == 4:
            a = a.squeeze(1)

        residual, _ = compute_weak_darcy_residual(u, a,
                                                  self.ranges,
                                                  f=self.f,
                                                  sigma_range=self.sigma_range,
                                                  test_fun=self.test_fun)

        residual = residual.pow(2).mean(dim=-1) * self.res_scaling

        res_bc = self.compute_boundary_constraint(u).mean()
        return residual + self.lam_bc * res_bc


    def compute_boundary_constraint(self, u, u_bc=0.):
        """
        Compute mean squared Dirichlet violation on all boundary nodes.

        Args:
            u: ``(B,H,W)`` or ``(B,1,H,W)``.
            u_bc: scalar boundary value.

        Returns:
            torch.Tensor of shape ``(B,)``.
        """
        if u.ndim == 4:  # [B,C,H,W]
            u = u.squeeze(1)

        # 2D boundary mask
        B, H, W = u.shape
        mask = torch.zeros(H, W, dtype=torch.bool, device=u.device)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        diff2 = (u - u_bc)[:, mask].pow(2)  # [B,#boundary]
        return diff2.mean(dim=-1) * self.bc_scaling


    def compute_residual_map(self, u, a, sigma_range=(2., 20.), test_fun='wd_wv'):
        """
        Compute test-indexed weak residual map reshaped to ``(B,H,W)``.

        Args:
            u: [B, H, H] - solution field
            a: [B, H, H] - diffusion coefficient field

        Returns:
            tuple ``(residual_map, coords)`` where residual map has shape
            ``(B,H,W)`` after squaring and scaling test residuals.
        """
        B, H, W = u.shape
        residual, coords = compute_weak_darcy_residual(u, a, self.ranges,
                                                  f=self.f, sigma_range=sigma_range, test_fun=test_fun)


        return residual.pow(2).view(B, H, W) * self.res_scaling, coords
