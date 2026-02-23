#!/usr/bin/env python3
"""
Entrypoint for PBFM pipeline:
1. Inverse pretraining (optional),
2. Residual-aware FM training with ConFIG (optional).

Config usage (keys accessed in this file)
-----------------------------------------
Top-level:
- `device`, `SEED`, `dataset`, `data_path`, `data_args`,
  `save_root_path`, `prefix_fm`, `prefix_inverse`, `do_fm`, `do_inverse`.
Nested:
- `flow_matching.residual.{type,args}`,
- `flow_matching.inverse_module.backbone.{type,args}`,
- `flow_matching.inverse_pretraining`,
- `flow_matching.backbone.{type,args}`,
- `flow_matching.training`.

Checkpoint paths:
- PBFM: `<save_root_path>/pbfm/<prefix_fm>_pbfm.pt`
- inverse: `<save_root_path>/pbfm/<prefix_inverse>_pbfm_inverse.pt`
"""

import os
import json
import random
import numpy as np
import argparse

import torch

from utils.util import DotDict, load_model
from utils.load_am_model import get_backbone, get_dataset, get_residual
from models.flow_matching import FlowMatchingModel

from training.pbfm_trainer import PBFMTrainer, pretrain_inverse



def _print_stage(title: str) -> None:
    """Print a visual stage header to stdout."""
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def train(config_path: str) -> None:
    """Execute configured PBFM stages with reproducibility seeding."""
    # ----- Load config -----
    cfg = DotDict(json.load(open(config_path)))
    device = cfg.device

    root_path = cfg.save_root_path
    prefix_fm = cfg.prefix_fm
    prefix_inverse = cfg.prefix_inverse

    fm_folder = os.path.join(root_path, "pbfm")
    fm_path = os.path.join(fm_folder, prefix_fm + "_pbfm.pt")

    inverse_folder = os.path.join(root_path, "pbfm")
    inverse_path = os.path.join(inverse_folder, prefix_inverse + "_pbfm_inverse.pt")

    do_fm = cfg.do_fm
    do_inverse_pre = cfg.do_inverse

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
    residual_type = cfg.flow_matching.residual.type
    residual_args = cfg.flow_matching.residual.args
    residual_model = get_residual(residual_type, data, residual_args)
    if isinstance(residual_model, torch.nn.Module):
        residual_model = residual_model.to(device)


    # ----- Inverse pretraining -----
    if do_inverse_pre:
        _print_stage("Stage 2: Inverse module pretraining")
        if os.path.isfile(inverse_path):
            raise ValueError("File already exists, will not start training.")

        backbone_inverse_type = cfg.flow_matching.inverse_module.backbone.type
        backbone_inverse_args = cfg.flow_matching.inverse_module.backbone.args
        backbone_inverse = get_backbone(backbone_inverse_type, backbone_inverse_args).to(device)


        pretrain_inverse(
            data,
            backbone_inverse,
            residual_model,
            inverse_folder,
            prefix_inverse + "_pbfm_inverse",
            cfg,
            **cfg.flow_matching.inverse_pretraining,
        )

        del backbone_inverse

    # ----- FM training -----
    if do_fm:
        _print_stage("Loading dataset")

        _print_stage("Stage 1: Flow Matching training")
        if os.path.isfile(fm_path):
            raise ValueError("File already exists, will not start training.")

        backbone_type = cfg.flow_matching.backbone.type
        backbone_args = cfg.flow_matching.backbone.args
        fm_backbone = get_backbone(backbone_type, backbone_args)

        backbone_inverse_type = cfg.flow_matching.inverse_module.backbone.type
        backbone_inverse_args = cfg.flow_matching.inverse_module.backbone.args
        backbone_inverse = get_backbone(backbone_inverse_type, backbone_inverse_args).to(device)
        backbone_inverse = load_model(backbone_inverse, inverse_path)

        fm_model = FlowMatchingModel(fm_backbone, device, **data_args)
        fm_train_args = cfg.flow_matching.training
        fm_trainer = PBFMTrainer(fm_model,
                                 data,
                                 backbone_inverse,
                                 residual_model,
                                 device, **fm_train_args)

        fm_trainer.train(fm_folder, prefix_fm + "_pbfm", cfg, verbose=True)


    _print_stage("Training complete")


def main() -> None:
    """CLI wrapper for `train`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
