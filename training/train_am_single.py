#!/usr/bin/env python3
"""
Entrypoint for single-model (non-joint) adjoint matching fine-tuning.

Config usage (keys accessed in this file)
-----------------------------------------
Top-level:
- `device`, `SEED`, `dataset`, `data_path`, `data_args`,
  `save_root_path`, `prefix_fm`, `prefix_inverse`, `prefix_am`,
  `do_am`.
Nested:
- `flow_matching.backbone.{type,args}`,
- `adjoint_matching.residual.{type,args}`,
- `adjoint_matching.use_ema_weights`,
- `adjoint_matching.inverse_module.backbone.{type,args}`,
- `adjoint_matching.training`.

Checkpoint paths:
- FM: `<save_root_path>/flow_matching/<prefix_fm>_fm.pt`
- inverse: `<save_root_path>/inverse/<prefix_inverse>_inverse.pt`
- AM: `<save_root_path>/adjoint_matching/<prefix_am>_am.pt`
"""

import os
import json
import random
import numpy as np
import argparse

import torch

from utils.util import DotDict, load_model
from utils.load_am_model import get_backbone, get_backbone_finetune, get_dataset, get_residual
from models.flow_matching import FlowMatchingModel
from models.adjoint_matching import AdjointMatchingModel
from training.adjoint_matching_trainer import AdjointMatchingTrainer



def _print_stage(title: str) -> None:
    """Print a visual stage header to stdout."""
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def train(config_path: str) -> None:
    """Run residual setup and optional single adjoint-matching fine-tuning."""
    # ----- Load config -----
    cfg = DotDict(json.load(open(config_path)))
    device = cfg.device

    root_path = cfg.save_root_path
    prefix_fm = cfg.prefix_fm
    prefix_inverse = cfg.prefix_inverse
    prefix_am = cfg.prefix_am

    fm_folder = os.path.join(root_path, "flow_matching")
    fm_path = os.path.join(fm_folder, prefix_fm + "_fm.pt")

    inverse_folder = os.path.join(root_path, "inverse")
    inverse_path = os.path.join(inverse_folder, prefix_inverse + "_inverse.pt")

    am_folder = os.path.join(root_path, "adjoint_matching")
    am_path = os.path.join(am_folder, prefix_am + "_am.pt")

    do_am = cfg.do_am

    data_args = cfg.data_args

    # SEED everything
    SEED = cfg.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Dataset
    _print_stage("Loading dataset")
    data = get_dataset(cfg.dataset, cfg.data_path)

    # ----- Residual model -----
    _print_stage("Preparing residual model")
    residual_type = cfg.adjoint_matching.residual.type
    residual_args = cfg.adjoint_matching.residual.args
    residual_model = get_residual(residual_type, data, residual_args)
    if isinstance(residual_model, torch.nn.Module):
        residual_model = residual_model.to(device)

    use_ema = cfg.adjoint_matching.use_ema_weights

    # ----- Adjoint matching (fine-tuning) -----
    if do_am:
        _print_stage("Stage 3: Adjoint Matching fine-tuning")
        if os.path.isfile(am_path):
            raise ValueError("File already exists, will not start training.")

        # base FM (for sampling)
        fm_backbone_type = cfg.flow_matching.backbone.type
        fm_backbone_args = cfg.flow_matching.backbone.args
        fm_backbone = get_backbone(fm_backbone_type, fm_backbone_args)
        fm_backbone = load_model(fm_backbone, fm_path, use_ema=use_ema)
        fm_model = FlowMatchingModel(fm_backbone, device, **data_args)

        # inverse backbone (pretrained)
        backbone_inverse_type = cfg.adjoint_matching.inverse_module.backbone.type
        backbone_inverse_args = cfg.adjoint_matching.inverse_module.backbone.args
        backbone_inverse = get_backbone(backbone_inverse_type, backbone_inverse_args)
        backbone_inverse = load_model(backbone_inverse, inverse_path)


        # base FM again (as finetune base)
        finetune_base_model = get_backbone(fm_backbone_type, fm_backbone_args)
        finetune_base_model = load_model(finetune_base_model, fm_path, use_ema=use_ema)


        # there is no extra fine-tuning backbone here
        am_model = AdjointMatchingModel(
            fm_model,
            finetune_base_model,
            backbone_inverse,
            device,
        )

        am_trainer = AdjointMatchingTrainer(
            am_model,
            residual_model,
            device,
            **cfg.adjoint_matching.training,
        )

        am_trainer.finetune(am_folder, prefix_am + "_am", cfg, verbose=True)

    _print_stage("Training complete")


def main() -> None:
    """CLI wrapper for `train`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
