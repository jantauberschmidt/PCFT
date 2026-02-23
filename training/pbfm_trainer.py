"""
Penalty-balanced flow-matching trainer using ConFIG gradient combination.

Per batch, the code computes FM loss and residual penalty gradients separately
and combines them with ConFIG before the optimizer step.
"""

from contextlib import nullcontext
import json
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from conflictfree.grad_operator import ConFIG_update_double

from utils.util import EmaWeights, save_model

def _worker_init_fn(_):
    """Limit each data-loader worker to one intra-op thread."""
    torch.set_num_threads(1)

class PBFMTrainer:
    """
    Train FM model with residual-aware conflict-free gradient updates.

    Objective
    ---------
    Scalar terms computed each batch:
    - ``L_FM = MSE(v_theta(x_t,t), target)``
    - ``L_res = mean(t * residual(x_1_pred, alpha_pred))``

    Gradients are not added as ``L_FM + lambda L_res``. Instead:
    ``g = ConFIG_update_double(grad L_FM, grad L_res)`` (or ``grad L_FM`` if
    residual grad has NaNs), and `g` is written to `.grad` for AdamW.

    Inputs/Outputs
    --------------
    - Expects data batches of shape ``(B, ...)``.
    - `train_epoch` returns mean FM loss used for logging.
    - `train` saves checkpoints every epoch and fixed snapshots every
      `save_every` epochs.
    """
    def __init__(self, flow_matching_model,
                 data,
                 inverse_model,
                 residual_model,
                 device,
                 batch_size=256,
                 lr=1e-4,
                 weight_decay=0.01,
                 p_warmup=0.01,
                 ema_decay=0.9998,
                 n_epochs=500,
                 n_workers_loader=4,
                 save_every=50):

        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.p_warmup = p_warmup
        self.batch_size = batch_size
        self.data = data
        self.n_workers_loader = n_workers_loader
        self.save_every = save_every

        self.data = data
        self.data_loader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=n_workers_loader,
            shuffle=True,
            pin_memory=True if n_workers_loader>0 else False,
            persistent_workers=True if n_workers_loader>0 else False,
            prefetch_factor=4 if n_workers_loader>0 else None,
            worker_init_fn=_worker_init_fn,
        )

        if torch.cuda.is_available():
            self.device_type = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.set_float32_matmul_precision("high")

        elif torch.backends.mps.is_available():
            self.device_type = "mps"

        else:
            self.device_type = "cpu"


        self.fm_model = flow_matching_model
        self.inverse_model = inverse_model
        self.residual_model = residual_model

        self.optim_fm = torch.optim.AdamW(self.fm_model.backbone.parameters(),
                                          lr, weight_decay=weight_decay)


        self.total_steps = max(1, self.n_epochs * len(self.data_loader))
        self.warmup_steps = max(1, int(self.p_warmup * self.total_steps))
        self._global_step = 0

        self.ema = None
        if ema_decay is not None:
            self.ema = EmaWeights(self.fm_model.backbone, decay=ema_decay)


    def get_ema_state(self):
        """Return EMA state dict if enabled, else ``None``."""
        return None if self.ema is None else self.ema.state_dict()


    def _set_lr_for_step(self):
        """Apply linear learning-rate warmup over initial update steps."""
        if self._global_step < self.warmup_steps:
            mult = float(self._global_step + 1) / float(self.warmup_steps)
            lr = self.lr * mult
        else:
            lr = self.lr
        for g in self.optim_fm.param_groups:
            g["lr"] = lr


    def compute_fm_target(self, data, t):
        """Construct interpolation state and FM target velocity."""
        noise = self.fm_model.sample_noise(data.shape[0])
        interpolation = (1 - t) * noise + t * data
        target = data - noise
        return interpolation, target


    def compute_loss(self, x_t, target, t, n_steps):
        """Compute FM loss and residual penalty estimate for one batch.

        `alpha1_pred` is obtained under `torch.no_grad()`. Residual remains
        differentiable w.r.t. `x1_pred`.
        """
        vt_pred = self.fm_model(x_t, t)

        t1 = t.clone()
        dt = (1 - t) / n_steps
        x1_pred = x_t + dt * vt_pred
        for _ in range(1, n_steps):
            t1 = t1 + dt
            vt_1 = self.fm_model(x1_pred, t1)
            x1_pred = x1_pred + dt * vt_1

        with torch.no_grad():
            alpha1_pred = self.inverse_model(x1_pred)
        residual = (t * self.residual_model.compute_residual(x1_pred, alpha1_pred)).mean()

        loss = F.mse_loss(vt_pred, target, reduction="mean")

        return loss, residual

    def train_epoch(self, epoch):
        """Run one epoch of ConFIG-based FM optimization."""
        self.fm_model.train()
        running_loss = []

        for data in self.data_loader:
            self._set_lr_for_step()

            B, *dims = data.size()
            data = data.to(self.device, non_blocking=True)

            t = torch.rand([B] + [1] * len(dims), device=self.device, dtype=data.dtype)
            interpolation, target = self.compute_fm_target(data, t)

            n_steps = epoch // (self.n_epochs // 4) + 1  # heuristic taken from original paper
            loss, residual = self.compute_loss(interpolation, target, t, n_steps=n_steps)

            # ConFIG step
            self.optim_fm.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)

            grad_fm_real, meta = get_config_grad_vector(self.fm_model.backbone)

            self.optim_fm.zero_grad(set_to_none=True)
            residual.backward()

            grad_res_real, meta2 = get_config_grad_vector(self.fm_model.backbone)

            assert meta == meta2, "Meta info mismatch between FM and residual grads"

            if torch.isnan(grad_res_real).any():
                g_config_real = grad_fm_real
            else:
                # ConFIG sees *real* gradients, so norm() etc. work
                g_config_real = ConFIG_update_double(grad_fm_real, grad_res_real)

            # Write conflict-free gradient back into .grad (complex restored)
            set_config_grad_vector(self.fm_model.backbone, g_config_real, meta)

            self.optim_fm.step()


            if self.ema is not None:
                self.ema.update(self.fm_model.backbone)

            self._global_step += 1
            running_loss.append(float(loss.detach()))

        return sum(running_loss) / max(1, len(running_loss))


    def train(self, save_path, save_name, cfg, verbose=True):
        """Run epoch loop with JSON logging and checkpoint writes."""
        for epoch in range(self.n_epochs):
            loss = self.train_epoch(epoch)

            if verbose:
                log_data = {"epoch": epoch,
                            "loss_fm": loss,
                            "lr": self.optim_fm.param_groups[0]["lr"]}
                print(json.dumps(log_data))

            # save (and overwrite) snapshot every epoch
            save_model(save_path, save_name, self.fm_model.backbone, self.optim_fm,
                       cfg, epoch=epoch,
                       ema_state_dict=self.get_ema_state())

            # save fixed snapshots for fraction of episodes
            if (epoch + 1) % self.save_every == 0:
                save_model(save_path, f"{save_name}_{epoch+1}",
                           self.fm_model.backbone, self.optim_fm,
                           cfg, epoch=epoch,
                           ema_state_dict=self.get_ema_state())


def pretrain_inverse(pretrain_data,
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
                     verbose=True
                     ):
    """Pretrain inverse module by minimizing residual on provided samples."""

    assert n_data % batch_size==0, 'n_data must be divisible by batch_size'
    optim = torch.optim.AdamW(inverse_backbone.parameters(), lr,
                              weight_decay=weight_decay)


    n = len(pretrain_data)
    n_epochs_adj = int(n_data / n * n_epochs)

    data_loader = DataLoader(pretrain_data, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs_adj):
        loss_log = []
        for x in data_loader:
            optim.zero_grad()

            x = x.to(cfg.device)
            alpha = inverse_backbone(x)

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


########  Helpers for complex gradients  ##########################

def get_config_grad_vector(module: torch.nn.Module) -> tuple[torch.Tensor, list[tuple[int, bool]]]:
    """
    Flatten all parameter grads of `module` into a single *real* vector.

    Complex grads are mapped via view_as_real(..., 2) -> R^{2n}.

    Returns
    -------
    flat_grad : torch.Tensor
        1D real tensor containing all grads (real + imag stacked).
    meta : list of (numel, is_complex)
        For each parameter with non-None grad, stores its original numel
        and whether it was complex.
    """
    flat_grads = []
    meta: list[tuple[int, bool]] = []

    for p in module.parameters():
        if p.grad is None:
            continue

        if p.grad.is_complex():
            g = torch.view_as_real(p.grad).contiguous().view(-1)  # length = 2 * p.numel()
            meta.append((p.numel(), True))
        else:
            g = p.grad.contiguous().view(-1)
            meta.append((p.numel(), False))

        flat_grads.append(g)

    if not flat_grads:
        return torch.tensor([], device=next(module.parameters()).device), meta

    flat_grad = torch.cat(flat_grads, dim=0)
    return flat_grad, meta


def set_config_grad_vector(module: torch.nn.Module,
                           flat_grad: torch.Tensor,
                           meta: list[tuple[int, bool]]) -> None:
    """
    Inverse of get_config_grad_vector: unflatten `flat_grad` and write into p.grad.

    Assumes same parameter order and metadata from `get_config_grad_vector`.
    """
    offset = 0
    meta_iter = iter(meta)

    for p in module.parameters():
        if p.grad is None:
            continue

        numel, is_complex = next(meta_iter)

        if is_complex:
            # We stored 2 * numel real entries = (..., 2) view
            numel_real = 2 * numel
            g_real = flat_grad[offset:offset + numel_real]
            offset += numel_real

            g_real = g_real.view(*p.shape, 2)
            # Real+imag reconstructed as complex; assume float32 underneath.
            g_complex = torch.view_as_complex(g_real.to(dtype=torch.float32))
            p.grad.copy_(g_complex.to(dtype=p.grad.dtype, device=p.grad.device))
        else:
            g = flat_grad[offset:offset + numel]
            offset += numel
            p.grad.copy_(g.view_as(p.grad).to(dtype=p.grad.dtype, device=p.grad.device))

    assert offset == flat_grad.numel(), "Flat gradient length mismatch in set_config_grad_vector"
