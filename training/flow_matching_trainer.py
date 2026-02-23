"""
Trainer for base flow-matching model fitting.

The trainer minimizes mean-squared error between the model velocity field and
the linear-interpolation target velocity, with optional EMA tracking and
mixed-precision acceleration.
"""

from contextlib import nullcontext
import json
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from utils.util import EmaWeights, save_model, clip_grad_norm_safe_

def _worker_init_fn(_):
    """Limit each data-loader worker to one intra-op thread."""
    torch.set_num_threads(1)

class FlowMatchingTrainer:
    """
    Optimize a flow-matching backbone on dataset samples.

    Objective
    ---------
    For data sample ``x_1``, noise sample ``x_0``, and ``t ~ U[0,1]``, this
    implementation forms
    ``x_t = (1-t)x_0 + t x_1`` and target vector field
    ``u_t = x_1 - x_0``.
    It minimizes
    ``L_FM(theta) = E[ ||v_theta(x_t, t) - u_t||_2^2 ]``.

    Inputs/Outputs
    --------------
    - Input data loader yields tensors shaped ``(B, ...)`` on `self.device`.
    - `train_epoch` returns mean scalar FM loss for that epoch.
    - `train` writes checkpoints via `save_model` every epoch and fixed snapshots
      every `save_every` epochs, and prints JSON logs.
    """
    def __init__(self, flow_matching_model,
                 data,
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

        self.fm_model = flow_matching_model

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


    def _amp_cm(self):
        """Return autocast context manager when AMP is enabled."""
        return nullcontext() if (self.amp_dtype is None) else torch.amp.autocast(device_type=self.device_type,
                                                                             dtype=self.amp_dtype)

    def _set_lr_for_step(self):
        """Apply linear warmup schedule over the first `warmup_steps` updates."""
        if self._global_step < self.warmup_steps:
            mult = float(self._global_step + 1) / float(self.warmup_steps)
            lr = self.lr * mult
        else:
            lr = self.lr
        for g in self.optim_fm.param_groups:
            g["lr"] = lr


    def compute_fm_target(self, data, t):
        """Construct interpolation state and target velocity for FM training."""
        noise = self.fm_model.sample_noise(data.shape[0])
        interpolation = (1 - t) * noise + t * data
        target = data - noise
        return interpolation, target


    def train_epoch(self):
        """Run one epoch of FM optimization (forward, backward, optimizer, EMA)."""
        self.fm_model.train()
        running_loss = []

        for data in self.data_loader:
            self._set_lr_for_step()

            B, *dims = data.size()
            data = data.to(self.device, non_blocking=True)

            t = torch.rand([B] + [1] * len(dims), device=self.device, dtype=data.dtype)
            interpolation, target = self.compute_fm_target(data, t)


            with self._amp_cm():
                vt_pred = self.fm_model(interpolation, t)
                loss = F.mse_loss(vt_pred, target, reduction="mean")

            self.optim_fm.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim_fm)
                clip_grad_norm_safe_(self.fm_model.backbone.parameters(), 1.0)
                self.scaler.step(self.optim_fm)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_safe_(self.fm_model.backbone.parameters(), 1.0)
                self.optim_fm.step()

            if self.ema is not None:
                self.ema.update(self.fm_model.backbone)

            self._global_step += 1
            running_loss.append(float(loss.detach()))

        return sum(running_loss) / max(1, len(running_loss))


    def train(self, save_path, save_name, cfg, verbose=True):
        """Run all epochs, print metrics, and save checkpoints/snapshots."""
        for epoch in range(self.n_epochs):
            loss = self.train_epoch()

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
