"""
Adjoint-matching fine-tuning trainer for image/state trajectory flow models.

This trainer minimizes a residual-based penalty objective (not a reward
maximization objective) by backpropagating through an adjoint approximation
along sampled memoryless trajectories.
"""

from contextlib import nullcontext
import json
import numpy as np
import math
import torch
import os
from torch.amp import GradScaler

from utils.util import save_model, clip_grad_norm_safe_

class AdjointMatchingTrainer:
    """
    Fine-tune flow backbone (and optionally inverse module) via adjoint matching.

    Objective
    ---------
    Let ``r(x, alpha)`` be the scalar residual penalty returned by
    `self.residual_model.compute_residual`. The trainer forms adjoint states from
    gradients of ``r`` and minimizes sampled-time penalties of drift mismatch:

    ``L_AM = E_{i in S}[ || sigma_i^{-1}(b_i^ft - b_i^base) + sigma_i a_i ||_2^2 ]``,

    where ``S`` is a sampled subset of rollout time indices and
    ``b_i = v_i + (sigma_i^2/(2 eta_i))(v_i - x_i/(t_i+eps))``.
    If `freeze_inverse=False`, an additional inverse update step minimizes mean
    terminal residual through `r.backward()`.

    Sign convention
    ---------------
    Residuals are penalties to minimize. No negative-reward sign flip is used.

    Inputs/Outputs
    --------------
    - Uses `am_model.sample_memoryless_rollout(...)` for trajectory sampling.
    - `finetune_epoch` returns `(loss_am, residual_am)` averaged over rollouts.
    - `finetune` logs JSON metrics and saves epoch checkpoints/snapshots with
      `save_model`.
    """
    def __init__(self,
                 adjoint_matching_model,
                 residual_model,
                 device,
                 freeze_inverse=True,
                 lr=2e-5,
                 weight_decay=0.01,
                 n_epochs=200,
                 save_every=50,
                 batch_size=8,
                 n_rollouts=8,
                 use_checkpointing=False,
                 steps=100,
                 K=20,
                 qt_steps_end=0.8,
                 kappa_memoryless=0.0,
                 use_tilted_time=False,
                 q=0.9,
                 lam_x=1000.,
                 ):

        self.freeze_inverse = freeze_inverse
        self.device = device
        self.n_epochs = n_epochs
        self.save_every = save_every
        self.batch_size = batch_size
        self.n_rollouts = n_rollouts
        self.use_checkpointing = use_checkpointing
        self.steps = steps
        self.K = K
        self.qt_steps_end = qt_steps_end
        self.kappa_memoryless = kappa_memoryless
        self.use_tilted_time = use_tilted_time
        self.q = q
        self.lam_x = lam_x

        self.lct_x = 1.6 * lam_x ** 2

        if self.use_tilted_time:
            self.t_finetune = get_tilted_time(self.steps, self.device, self.q)
        else:
            self.t_finetune = torch.linspace(0., 1., self.steps + 1, device=self.device)


        if adjoint_matching_model.base_fm_model.backbone.__class__.__name__ == 'DiT':
            # fine-tuning DiT backbone with half precision leads to nan, therefore disable
            self.device_type = None
            self.amp_dtype = None
            self.scaler = None
        else:
            if torch.cuda.is_available():
                self.device_type = "cuda"
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                    self.scaler = None
                else:
                    self.amp_dtype = torch.float16
                    self.scaler = GradScaler(device="cuda")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.set_float32_matmul_precision("high")
            elif torch.backends.mps.is_available():
                self.device_type = "mps"
                self.amp_dtype = torch.float16
                self.scaler = None
            else:
                self.device_type = None
                self.amp_dtype = None
                self.scaler = None


        self.residual_model = residual_model

        self.am_model = adjoint_matching_model

        if self.freeze_inverse:
            self.params = self.am_model.backbone_finetune.parameters()
        else:
            self.params = (list(self.am_model.backbone_finetune.parameters()) +
                           list(self.am_model.backbone_inverse.parameters()))

        self.optim_am = torch.optim.AdamW(self.params,
                                          lr, weight_decay=weight_decay)


    def _amp_cm(self):
        """Return autocast context manager when AMP is active."""
        return nullcontext() if (self.amp_dtype is None) else torch.amp.autocast(device_type=self.device_type,
                                                                             dtype=self.amp_dtype)

    def _adjoint_loss_single_step(self, xt, t, sigma_i, eta_i, ax_i, eps=0.01):
        """Compute per-step adjoint matching loss contribution.

        Notes
        -----
        - `vt_x_base` is evaluated under `torch.no_grad()`.
        - `vt_finetune` remains differentiable and receives gradients.
        """

        sigma_safe = torch.clamp(sigma_i, min=1e-6)
        eta_safe = torch.clamp(eta_i, min=1e-6)


        with self._amp_cm():

            with torch.no_grad():
                vt_x_base = self.am_model.vt_x_base(xt, t).detach()
            vt_x = self.am_model.vt_finetune(xt, t)


            spatial_dims_x = tuple(range(1, vt_x.ndim))

            bt_x_ft = vt_x + (sigma_safe ** 2 / (2 * eta_safe)) * (vt_x - 1 / (t + eps) * xt)
            bt_x_base = vt_x_base + (sigma_safe ** 2 / (2 * eta_safe)) * (vt_x_base - 1 / (t + eps) * xt)

            loss_x = (1 / sigma_safe * (bt_x_ft - bt_x_base)
                      + sigma_safe * ax_i).pow(2).sum(dim=spatial_dims_x)


            loss_x = torch.clamp(loss_x, max=self.lct_x)

        return loss_x.sum().float()


    def lean_adjoint_ode(self, xt_traj, xt_base_traj, t_steps):
        """Integrate reverse-time adjoint dynamics with explicit Euler updates.

        Returns
        -------
        a_x : list[torch.Tensor]
            Adjoint tensors aligned with rollout time order.
        r_terminal : torch.Tensor
            Mean terminal residual term used for logging/inverse update.
        """

        x1 = xt_traj[-1].detach().clone().requires_grad_(True)

        with self._amp_cm():
            with torch.no_grad():
                alpha_1 = self.am_model.alpha_pred(x1)
            alpha_1_grad = self.am_model.alpha_pred(x1)
            if self.am_model.base_fm_model.latent_fm:
                # if latent, decode here. gradients wrt x will be pushed through vae.
                r1 = self.residual_model.compute_residual(self.am_model.base_fm_model.vae(x1), alpha_1)
                r1_grad = self.residual_model.compute_residual(self.am_model.base_fm_model.vae(x1.detach()).detach(), alpha_1_grad)

            else:
                r1 = self.residual_model.compute_residual(x1, alpha_1)
                r1_grad = self.residual_model.compute_residual(x1.detach(), alpha_1_grad)


        r1 = r1.float()


        # Compute gradients w.r.t. both x and alpha
        gx, = torch.autograd.grad(
            outputs=r1,
            inputs=x1,
            grad_outputs=torch.ones_like(r1),
            retain_graph=False,
            create_graph=False
        )

        # Residual is minimized as a penalty, so no sign inversion here.
        a_x0 = (self.lam_x * gx).detach()

        # Initialize adjoint state as concatenation [a_x, a_alpha]
        a_x = [a_x0]

        sigmas = self.am_model.base_fm_model.sigma_memoryless(t_steps, kappa=self.kappa_memoryless)
        etas = self.am_model.base_fm_model.eta(t_steps)

        for i, xt in enumerate(reversed(xt_traj)):

            i_t = -i - 1
            t = t_steps[i_t]
            h =  t_steps[i_t] -  t_steps[i_t - 1]
            xt = xt.clone().requires_grad_(True)
            sigma = sigmas[i]
            eta = etas[i]

            # with self._amp_cm():
            vt_x_base = self.am_model.vt_x_base(xt, t)  # depends on xt

            bt_x = vt_x_base + (sigma ** 2 / (2 * eta)) * (vt_x_base - 1 / (t + h) * xt)

            at_x = a_x[i]


            bt_x = bt_x.float()
            at_x = at_x.float()

            # VJP computation
            g_xt,  = torch.autograd.grad(
                outputs=bt_x,
                inputs=xt,
                grad_outputs=at_x,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )


            # depending on definition of network architectures, gradients can be none
            if g_xt is None: g_xt = torch.zeros_like(xt)

            # reverse-time Euler update
            at_x = at_x + h * g_xt

            a_x.append(at_x.detach())

            if i == (len(t_steps) - 2):
                break

        # Return adjoints for both variables
        return a_x[::-1], r1_grad.mean()

    def compute_adjoint_loss(self, x_traj, x_base_traj, t_steps):
        """Accumulate sampled-time adjoint loss and apply backward passes."""
        steps = len(t_steps)

        a_x, r = self.lean_adjoint_ode(x_traj, x_base_traj, t_steps)

        if self.K is None:
            sample_steps = np.arange(0, steps)
        else:
            qt75 = int(steps * self.qt_steps_end)
            last_steps = np.arange(qt75, steps)
            first_steps = np.random.choice(qt75, size=self.K)
            sample_steps = np.concatenate([first_steps, last_steps])

        sigmas = self.am_model.base_fm_model.sigma_memoryless(t_steps, kappa=self.kappa_memoryless)
        etas = self.am_model.base_fm_model.eta(t_steps)

        loss_norm = self.batch_size * self.n_rollouts * len(sample_steps)
        loss_log_accum = 0.0
        for i in sample_steps:
            xt = x_traj[i]
            t = t_steps[i]

            # Convert scalars to tensors for checkpointing
            sigma_i = sigmas[i]
            eta_i = etas[i]
            ax_i = a_x[i]

            if self.use_checkpointing:
                step_loss = torch.utils.checkpoint.checkpoint(
                    lambda *args: self._adjoint_loss_single_step(*args),
                    xt, t, sigma_i, eta_i, ax_i,
                    use_reentrant=False,
                )
            else:
                step_loss = self._adjoint_loss_single_step(xt, t, sigma_i, eta_i, ax_i)


            step_loss = step_loss / loss_norm
            loss_log_accum += float(step_loss.detach().cpu())
            if self.scaler is not None:
                self.scaler.scale(step_loss).backward()
            else:
                step_loss.backward()

        if not self.freeze_inverse:
            # update inverse predictor
            r.backward()

        return loss_log_accum, float(r.detach().cpu())


    def finetune_epoch(self):
        """Run one AM optimization epoch over `n_rollouts` sampled trajectories."""
        self.optim_am.zero_grad()
        loss_log = []
        residuals = []

        for _ in range(self.n_rollouts):
            (x, x_base, t_steps) = self.am_model.sample_memoryless_rollout(batch_size=self.batch_size,
                                                                       t_steps=self.t_finetune,
                                                                       kappa_memoryless=self.kappa_memoryless)

            loss, r = self.compute_adjoint_loss(x, x_base, t_steps)

            loss_log.append(loss)
            residuals.append(r)

        if self.scaler is not None:
            self.scaler.unscale_(self.optim_am)
            clip_grad_norm_safe_(self.params, 1.0)
            self.scaler.step(self.optim_am)
            self.scaler.update()
        else:
            clip_grad_norm_safe_(self.params, 1.0)
            self.optim_am.step()

        return sum(loss_log) / len(loss_log), sum(residuals) / len(residuals)


    def finetune(self, save_path, save_name, cfg, verbose=True):
        """Execute AM fine-tuning loop with JSON logging and checkpoint writes."""
        for epoch in range(self.n_epochs):

            loss, residual = self.finetune_epoch()

            if verbose:
                log_data = {"epoch": epoch,
                            "loss_am": loss,
                            "residual_am": residual,
                            "lr": self.optim_am.param_groups[0]["lr"]}
                print(json.dumps(log_data))

            # save (and overwrite) snapshot every epoch
            save_model(save_path, save_name, self.am_model.backbone_finetune, self.optim_am,
                       cfg, epoch=epoch,
                       ema_state_dict=None)

            if not self.freeze_inverse:
                inverse_folder = os.path.join(cfg.save_root_path, "inverse")
                inverse_save_name = f"{cfg.prefix_inverse}_ft_inverse"

                save_model(inverse_folder, inverse_save_name,
                           self.am_model.backbone_inverse, self.optim_am,
                           cfg, epoch=epoch,
                           ema_state_dict=None)

            # save fixed snapshots for fraction of episodes
            if (epoch + 1) % self.save_every == 0:
                save_model(save_path, f"{save_name}_{epoch + 1}",
                           self.am_model.backbone_finetune, self.optim_am,
                           cfg, epoch=epoch,
                           ema_state_dict=None)





def get_tilted_time(steps, device, q=0.9):
    """
    Return a monotone time grid tilted toward endpoints.

    ``t_ft = (1-q) t + q * 0.5*(1-cos(pi t))`` for uniform base grid ``t``.
    """
    t_base = torch.linspace(0., 1., steps + 1, device=device)

    g = 0.5 * (1.0 - torch.cos(math.pi * t_base))  # dense near 0 and 1
    t_ft = (1.0 - q) * t_base + q * g

    return t_ft




# move outside because full model is not needed for this
def pretrain_inverse(fm_model,
                     inverse_backbone,
                     residual,
                     save_path,
                     save_name,
                     cfg,
                     n_data=512,
                     n_epochs=1000,
                     batch_size=32,
                     lr=1e-4,
                     weight_decay=0.01,
                     steps_sampling=100,
                     verbose=True
                     ):
    """Pretrain inverse backbone on generated FM states by residual minimization.

    The function:
    1. generates fixed pretraining states with `fm_model.generate_pretrain_data`,
    2. predicts inverse parameters,
    3. minimizes mean residual `residual.compute_residual(..., pretrain=True)`.
    """

    assert n_data % batch_size==0, 'n_data must be divisible by batch_size'
    optim = torch.optim.AdamW(inverse_backbone.parameters(), lr,
                              weight_decay=weight_decay)

    # collect data and store on cpu
    pretrain_data = fm_model.generate_pretrain_data(n_data,
                                                    batch_size,
                                                    steps_sampling)

    print('Collected data for pretraining.')

    for epoch in range(n_epochs):
        loss_log = []
        for idx_start in range(0, n_data, batch_size):
            optim.zero_grad()
            x = pretrain_data[idx_start:idx_start + batch_size]
            x = x.to(fm_model.device)
            alpha = inverse_backbone(x)

            if fm_model.latent_fm:
                with torch.no_grad():
                    x = fm_model.vae(x)
            loss = residual.compute_residual(x.detach(), alpha, pretrain=True).mean()

            loss.backward()
            optim.step()
            loss_log.append(loss.detach().item())


        if verbose:
            log_data = {"epoch": epoch,
                        "residual_pretrain": sum(loss_log) / len(loss_log)}
            print(json.dumps(log_data))

        save_model(save_path, save_name, inverse_backbone, optim,
                   cfg, epoch=epoch,
                   ema_state_dict=None)


