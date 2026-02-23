"""
Joint adjoint-matching trainer for coupled state and parameter trajectories.

This variant fine-tunes the joint model using residual-penalty adjoint matching
for both state variable `x` and inverse/parameter variable `alpha`.
"""

from contextlib import nullcontext
import json
import numpy as np
import math
import torch
from torch.amp import GradScaler

from utils.util import save_model, clip_grad_norm_safe_

class JointAdjointMatchingTrainer:
    """
    Fine-tune joint AM model with adjoint penalties on `x` and `alpha`.

    Objective
    ---------
    With residual penalty ``r(x, alpha)`` at terminal time, the trainer computes
    initial adjoints from
    ``a_x(T)=lam_x * d r / d x`` and ``a_alpha(T)=lam_alpha * d r / d alpha``.
    It minimizes sampled-time mismatch penalties:

    ``L_joint = E_{i in S}[ ||sigma_i^{-1}(b_x^ft-b_x^base)+sigma_i a_x||_2^2
                           +||sigma_i^{-1}(b_a^ft-b_a^base)+sigma_i a_a||_2^2 ]``.

    A regularization gradient term may be added through `reg_scaling` inside
    reverse-time adjoint integration.

    Sign convention
    ---------------
    `residual_model.compute_residual` is treated as a penalty to minimize.

    Inputs/Outputs
    --------------
    - Uses `sample_memoryless_rollout_joint(...)` to obtain `(x, x_base, alpha, t)`.
    - `finetune_epoch` returns averaged `(loss_am, residual_am)`.
    - `finetune` writes checkpoints and emits JSON logs per epoch.
    """
    def __init__(self,
                 adjoint_matching_model,
                 residual_model,
                 device,
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
                 lam_alpha=1000.,
                 reg_scaling=0.1
                 ):

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
        self.lam_alpha = lam_alpha
        self.reg_scaling = reg_scaling

        self.lct_x = 1.6 * lam_x ** 2
        self.lct_alpha = 1.6 * lam_alpha ** 2

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

        self.params = self.am_model.backbone_finetune.parameters()
        self.optim_am = torch.optim.AdamW(self.params,
                                          lr, weight_decay=weight_decay)


    def _amp_cm(self):
        """Return autocast context manager when AMP is active."""
        return nullcontext() if (self.amp_dtype is None) else torch.amp.autocast(device_type=self.device_type,
                                                                             dtype=self.amp_dtype)

    def _adjoint_loss_single_step(self, xt, alpha_t, t, sigma_i, eta_i, ax_i, aalpha_i, eps=0.01):
        """Compute one sampled-time joint adjoint loss contribution.

        Base vector fields are detached; fine-tuned vector fields are
        differentiable and contribute gradients to trainable parameters.
        """

        sigma_safe = torch.clamp(sigma_i, min=1e-6)
        eta_safe = torch.clamp(eta_i, min=1e-6)


        with self._amp_cm():

            with torch.no_grad():
                vt_x_base = self.am_model.vt_x_base(xt, t).detach()
                vt_alpha_base = self.am_model.vt_alpha_base(xt, alpha_t, vt_x_base, t).detach()
            vt_x, vt_alpha = self.am_model.vt_finetune(xt, alpha_t, vt_alpha_base, t)


            spatial_dims_x = tuple(range(1, vt_x.ndim))

            bt_x_ft = vt_x + (sigma_safe ** 2 / (2 * eta_safe)) * (vt_x - 1 / (t + eps) * xt)
            bt_x_base = vt_x_base + (sigma_safe ** 2 / (2 * eta_safe)) * (vt_x_base - 1 / (t + eps) * xt)

            loss_x = (1 / sigma_safe * (bt_x_ft - bt_x_base)
                      + sigma_safe * ax_i).pow(2).sum(dim=spatial_dims_x)


            spatial_dims_alpha = tuple(range(1, vt_alpha.ndim))

            bt_alpha_ft = vt_alpha + (sigma_safe ** 2 / (2 * eta_safe)) * (vt_alpha - 1 / (t + eps) * alpha_t)
            bt_alpha_base = vt_alpha_base + (sigma_safe ** 2 / (2 * eta_safe)) * (vt_alpha_base - 1 / (t + eps) * alpha_t)

            loss_alpha = (1 / sigma_safe * (bt_alpha_ft - bt_alpha_base)
                      + sigma_safe * aalpha_i).pow(2).sum(dim=spatial_dims_alpha)

            loss_x = torch.clamp(loss_x, max=self.lct_x)
            loss_alpha = torch.clamp(loss_alpha, max=self.lct_alpha)

        return (loss_x + loss_alpha).sum().float()


    def lean_adjoint_ode(self, xt_traj, xt_base_traj, alpha_traj, t_steps):
        """Run reverse-time explicit-Euler adjoint integration for `(x, alpha)`."""

        x1 = xt_traj[-1].clone().requires_grad_(True)
        alpha_1 = alpha_traj[-1].clone().requires_grad_(True)

        with self._amp_cm():
            if self.am_model.base_fm_model.latent_fm:
                # if latent, decode here. gradients wrt x will be pushed through vae.
                r1 = self.residual_model.compute_residual(self.am_model.base_fm_model.vae(x1), alpha_1)
            else:
                r1 = self.residual_model.compute_residual(x1, alpha_1)

        r1 = r1.float()

        # Compute gradients w.r.t. both x and alpha
        gx, ga = torch.autograd.grad(
            outputs=r1,
            inputs=(x1, alpha_1),
            grad_outputs=torch.ones_like(r1),
            retain_graph=False,
            create_graph=False
        )

        # Residual is a penalty objective, so no sign inversion.
        a_x0 = (self.lam_x * gx).detach()
        a_alpha0 = (self.lam_alpha * ga).detach()

        # Initialize adjoint state as concatenation [a_x, a_alpha]
        a_x = [a_x0]
        a_alpha = [a_alpha0]

        sigmas = self.am_model.base_fm_model.sigma_memoryless(t_steps, kappa=self.kappa_memoryless)
        etas = self.am_model.base_fm_model.eta(t_steps)

        for i, (xt, xt_base, alpha_t) in enumerate(zip(reversed(xt_traj),
                                                       reversed(xt_base_traj),
                                                       reversed(alpha_traj))):

            i_t = -i - 1
            t = t_steps[i_t]
            h =  t_steps[i_t] -  t_steps[i_t - 1]
            xt = xt.clone().requires_grad_(True)
            alpha_t = alpha_t.clone().requires_grad_(True)
            sigma = sigmas[i]
            eta = etas[i]

            # with self._amp_cm():
            vt_x_base = self.am_model.vt_x_base(xt, t)  # depends on xt
            vt_alpha_base = self.am_model.vt_alpha_base(xt, alpha_t, vt_x_base, t)  # depends on xt, alpha_t

            bt_x = vt_x_base + (sigma ** 2 / (2 * eta)) * (vt_x_base - 1 / (t + h) * xt)
            bt_alpha = vt_alpha_base  + (sigma ** 2 / (2 * eta)) * (vt_alpha_base - 1 / (t+h) * alpha_t)

            at_x = a_x[i]
            at_alpha = a_alpha[i]

            # regularization term (to base flow)
            if self.reg_scaling > 0.:
                with torch.no_grad():
                    vt_x_base_base = self.am_model.vt_x_base(xt_base, t)
                    # Build reference alpha drift on base trajectory for regularization.
                    vt_alpha_reg = self.am_model.vt_alpha_base(xt_base, alpha_t, vt_x_base_base, t)

                _, vt_alpha_fine = self.am_model.vt_finetune(xt, alpha_t, vt_alpha_base, t)
                spatial_dims = tuple(range(1, vt_alpha_fine.ndim))  # (1,2) for [B,H,W]
                reg_term = self.reg_scaling * (vt_alpha_fine - vt_alpha_reg).pow(2).mean(dim=spatial_dims)

            else:
                reg_term = torch.tensor(0.0, device=self.device).requires_grad_(True)

            bt_x = bt_x.float()
            bt_alpha = bt_alpha.float()
            reg_term = reg_term.float()
            at_x = at_x.float()
            at_alpha = at_alpha.float()

            # VJP computation
            g_xt, g_alpha = torch.autograd.grad(
                outputs=(bt_x, bt_alpha, reg_term),
                inputs=(xt, alpha_t),
                grad_outputs=(at_x, at_alpha, torch.ones_like(reg_term)),
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )


            # depending on definition of network architectures, gradients can be none
            if g_xt is None: g_xt = torch.zeros_like(xt)
            if g_alpha is None: g_alpha = torch.zeros_like(alpha_t)

            # reverse-time Euler update
            at_x = at_x + h * g_xt
            at_alpha = at_alpha + h * g_alpha

            a_x.append(at_x.detach())
            a_alpha.append(at_alpha.detach())

            if i == (len(t_steps) - 2):
                break

        # Return adjoints for both variables
        return (a_x[::-1], a_alpha[::-1]), r1.mean().detach().cpu()

    def compute_adjoint_loss(self, x_traj, x_base_traj, alpha_traj, t_steps):
        """Accumulate sampled joint adjoint losses and backpropagate."""
        steps = len(t_steps)

        (a_x, a_alpha), r = self.lean_adjoint_ode(x_traj, x_base_traj, alpha_traj, t_steps)

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
            alpha_t = alpha_traj[i]
            t = t_steps[i]

            # Convert scalars to tensors for checkpointing
            sigma_i = sigmas[i]
            eta_i = etas[i]
            ax_i = a_x[i]
            aalpha_i = a_alpha[i]

            if self.use_checkpointing:
                step_loss = torch.utils.checkpoint.checkpoint(
                    lambda *args: self._adjoint_loss_single_step(*args),
                    xt, alpha_t, t, sigma_i, eta_i, ax_i, aalpha_i,
                    use_reentrant=False,
                )
            else:
                step_loss = self._adjoint_loss_single_step(xt, alpha_t, t, sigma_i, eta_i, ax_i, aalpha_i,)


            step_loss = step_loss / loss_norm
            loss_log_accum += float(step_loss.detach().cpu())
            if self.scaler is not None:
                self.scaler.scale(step_loss).backward()
            else:
                step_loss.backward()


        return loss_log_accum, float(r.detach().cpu())


    def finetune_epoch(self):
        """Run one optimization epoch over multiple joint rollouts."""
        self.optim_am.zero_grad()
        loss_log = []
        residuals = []

        for _ in range(self.n_rollouts):
            (x, x_base,
             alpha, t_steps) = self.am_model.sample_memoryless_rollout_joint(batch_size=self.batch_size,
                                                                             t_steps=self.t_finetune,
                                                                             kappa_memoryless=self.kappa_memoryless)

            loss, r = self.compute_adjoint_loss(x, x_base, alpha, t_steps)

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
        """Execute full fine-tuning loop with logging and checkpointing."""
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

            # save fixed snapshots for fraction of episodes
            if (epoch + 1) % self.save_every == 0:
                save_model(save_path, f"{save_name}_{epoch + 1}",
                           self.am_model.backbone_finetune, self.optim_am,
                           cfg, epoch=epoch,
                           ema_state_dict=None)



def get_tilted_time(steps, device, q=0.9):
    """
    Return tilted time discretization emphasizing endpoint resolution.
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
    """Pretrain inverse backbone by minimizing residual on generated FM states."""

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


