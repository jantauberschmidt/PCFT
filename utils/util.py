import time
import contextlib
import os
import numpy as np
import math
import torch
import torch.nn as nn
from typing import Any, Callable, Tuple, Optional, Dict, Iterable, Type
import inspect
import warnings


def _collect_excluded_prefixes(root: nn.Module, flag_name: str = "_exclude_from_saving") -> Tuple[str, ...]:
    if getattr(root, flag_name, False):
        return ("",)  # exclude everything under root
    return tuple(
        name + "."
        for name, mod in root.named_modules()
        if name and getattr(mod, flag_name, False)
    )

def _startswith_any(key: str, prefixes: Tuple[str, ...]) -> bool:
    return any(key.startswith(pfx) for pfx in prefixes)

def _filter_state_dict(sd: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    if not prefixes:
        return sd
    if "" in prefixes:
        return {}
    return {k: v for k, v in sd.items() if not _startswith_any(k, prefixes)}

def save_model(
    save_path: str,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: Optional[int] = None,
    ema_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    prefixes = _collect_excluded_prefixes(model)

    model_sd = _filter_state_dict(model.state_dict(), prefixes)
    if ema_state_dict is not None:
        ema_state_dict = _filter_state_dict(ema_state_dict, prefixes)

    checkpoint = {
        "epoch": epoch,
        "model": model_sd,
        "optim": optimizer.state_dict(),
        "config": config,
        "ema": ema_state_dict,
    }
    torch.save(checkpoint, os.path.join(save_path, f"{model_name}.pt"))


def _strict_diff_ignoring_exclusions(
    model: nn.Module,
    sd_in: Dict[str, torch.Tensor],
    excluded: Tuple[str, ...],
    check_dtype: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """
    Returns (missing, unexpected, shape_or_dtype_mismatch), all w.r.t. keys NOT excluded.
    """
    model_sd = model.state_dict()
    allowed_keys = tuple(k for k in model_sd.keys() if not _startswith_any(k, excluded))

    # What should be present
    should = set(allowed_keys)
    # What we plan to load
    have = set(k for k in sd_in.keys() if not _startswith_any(k, excluded))

    missing = tuple(sorted(should - have))
    unexpected = tuple(sorted(have - should))

    # Shape/dtype mismatches on intersection
    mismatches = []
    for k in sorted(should & have):
        v_in = sd_in[k]
        v_ref = model_sd[k]
        if v_in.shape != v_ref.shape:
            mismatches.append(k)
            continue
        if check_dtype and (v_in.dtype != v_ref.dtype):
            mismatches.append(k)

    return missing, unexpected, tuple(mismatches)


def load_model(
    model: nn.Module,
    path: str,
    use_ema: bool = False,
    flag_name: str = "_exclude_from_saving",
    check_dtype: bool = True,
    ckpt = None
) -> nn.Module:
    """
    Strict loading w.r.t. non-excluded parameters/buffers.
    Excluded submodules (marked via `flag_name`) are ignored—no errors raised for them.

    Raises:
        RuntimeError with a readable diff if there are missing/unexpected/mismatch keys
        outside excluded prefixes.
    """
    if ckpt is None:
        ckpt = torch.load(path, weights_only=False, map_location="cpu")

    key = "ema" if use_ema else "model"
    if key not in ckpt or ckpt[key] is None:
        raise RuntimeError(f"Checkpoint missing '{key}' state_dict.")

    sd_in: Dict[str, torch.Tensor] = ckpt[key]
    excluded = _collect_excluded_prefixes(model, flag_name=flag_name)

    # Pre-check strictness (ignoring excluded prefixes)
    missing, unexpected, mismatches = _strict_diff_ignoring_exclusions(
        model, sd_in, excluded, check_dtype=check_dtype
    )
    if missing or unexpected or mismatches:
        msgs = []
        if missing:
            msgs.append(f"Missing keys (non-excluded): {missing}")
        if unexpected:
            msgs.append(f"Unexpected keys (non-excluded): {unexpected}")
        if mismatches:
            msgs.append(f"Shape/dtype mismatches (non-excluded): {mismatches}")
        raise RuntimeError("Strict load failed:\n" + "\n".join(msgs))

    # Filter incoming state_dict to non-excluded keys only
    sd_filtered = _filter_state_dict(sd_in, excluded)

    # Load with strict=False (we already enforced strictness for allowed keys)
    missing_after, unexpected_after = model.load_state_dict(sd_filtered, strict=False)
    # PyTorch returns tuples of keys; they may include only excluded prefixes here.
    # Sanity: ensure any reported keys are indeed excluded.
    if missing_after:
        non_excluded_missing = tuple(k for k in missing_after if not _startswith_any(k, excluded))
        if non_excluded_missing:
            raise RuntimeError(f"Unexpected missing after load (non-excluded): {non_excluded_missing}")
    if unexpected_after:
        non_excluded_unexpected = tuple(k for k in unexpected_after if not _startswith_any(k, excluded))
        if non_excluded_unexpected:
            raise RuntimeError(f"Unexpected unexpected after load (non-excluded): {non_excluded_unexpected}")

    return model


class DotDict(dict):
    """Recursively turns a dict into attribute‑accessible object."""
    def __getattr__(self, item):
        try:
            value = self[item]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)     # recurse once
                self[item] = value         # cache
            return value
        except KeyError:
            raise AttributeError(item) from None

    # make tab‑completion nicer in some IDEs
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def strict_kwargs(kwargs: dict, cls_or_func):
    """
    • Asserts that EVERY parameter in `cls_or_func`'s signature
      (except 'self', *args and **kwargs) is provided in `kwargs`.
    • Asserts that `kwargs` contains NO extraneous keys.

    Returns `kwargs` unchanged so it can be used inline:

        model = MyModel(**strict_kwargs(cfg.model, MyModel))
    """
    sig = inspect.signature(cls_or_func).parameters

    # drop parameters we never pass explicitly
    valid = {
        name for name, par in sig.items()
        if name != "self" and par.kind not in
           (inspect.Parameter.VAR_POSITIONAL,  # *args
            inspect.Parameter.VAR_KEYWORD)     # **kwargs
    }

    unknown = [k for k in kwargs if k not in valid]
    if unknown:
        warnings.warn(
            f"{cls_or_func.__name__}: unexpected argument(s) {unknown}"
        )

    missing = [k for k in valid if k not in kwargs]
    if missing:
        warnings.warn(
            f"{cls_or_func.__name__}: config missing required argument(s) {missing}"
        )

    return kwargs


class EmaWeights:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point or p.is_complex():
                self.shadow[name] = p.detach().clone()


    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(d).add_(msd[k], alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> dict:
        return {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        msd = model.state_dict()
        for k, v in backup.items():
            msd[k].copy_(v)


@torch.no_grad()
def clip_grad_norm_safe_(parameters, max_norm: float, eps: float = 1e-12) -> float:
    """
    Global L2 grad clipping that supports real, complex, and sparse gradients.
    Keeps reductions on-device; performs at most one host sync.
    Returns the total grad norm as a Python float.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    # Accumulate squared norms on the grads' device without host syncs
    sq_terms = []
    for p in params:
        g = p.grad
        if g.is_sparse:
            g = g.coalesce().values()
        if torch.is_complex(g):
            sq_terms.append((g.real.square() + g.imag.square()).sum())
        else:
            # cast to float to avoid fp16 overflow in sum
            sq_terms.append(g.float().square().sum())

    total_sq = torch.stack(sq_terms).sum()
    total_norm = total_sq.sqrt()  # still on device

    # Single sync here (same as stock impl when comparing to threshold)
    total = float(total_norm.item())

    # Compute clipping coeff on device; one branch, applied to all params
    scale = (max_norm / (total_norm + eps)).clamp(max=1.0)
    if scale < 1.0:  # compares a 0-dim tensor; incurs no extra sync after .item() above
        for p in params:
            p.grad.mul_(scale)

    return total

