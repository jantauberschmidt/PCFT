#!/usr/bin/env python3
"""
Entrypoint for 3-stage joint AM pipeline:
1. Base flow matching training (optional),
2. Inverse-module pretraining (optional),
3. Joint adjoint matching fine-tuning (optional).

Config usage (keys accessed in this file)
-----------------------------------------
Top-level:
- `device`, `SEED`, `dataset`, `data_path`, `data_args`,
  `save_root_path`, `prefix_fm`, `prefix_inverse`, `prefix_am`,
  `do_fm`, `do_inverse`, `do_am`.
Nested:
- `flow_matching.backbone.{type,args}`, `flow_matching.training`,
- `adjoint_matching.residual.{type,args}`,
- `adjoint_matching.use_ema_weights`,
- `adjoint_matching.inverse_module.backbone.{type,args}`,
- `adjoint_matching.inverse_pretraining`,
- `adjoint_matching.model_finetune.backbone.{type,args}`,
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
from models.joint_adjoint_matching import JointAdjointMatchingModel
from training.flow_matching_trainer import FlowMatchingTrainer
from training.joint_adjoint_matching_trainer import JointAdjointMatchingTrainer, pretrain_inverse



def _print_stage(title: str) -> None:
    """Print a visual stage header to stdout."""
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def train(config_path: str) -> None:
    """Execute configured training stages in order.

    Notes
    -----
    - If a target checkpoint path already exists for a stage, this script raises
      and aborts instead of overwriting.
    - RNG seeding is applied to Python, NumPy, and PyTorch (including CUDA).
    """
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

    do_fm = cfg.do_fm
    do_inverse_pre = cfg.do_inverse
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

    # ----- FM training -----
    if do_fm:
        _print_stage("Stage 1: Flow Matching training")
        if os.path.isfile(fm_path):
            raise ValueError("File already exists, will not start training.")

        backbone_type = cfg.flow_matching.backbone.type
        backbone_args = cfg.flow_matching.backbone.args
        fm_backbone = get_backbone(backbone_type, backbone_args)

        fm_model = FlowMatchingModel(fm_backbone, device, **data_args)
        fm_train_args = cfg.flow_matching.training
        fm_trainer = FlowMatchingTrainer(fm_model, data, device, **fm_train_args)

        fm_trainer.train(fm_folder, prefix_fm + "_fm", cfg, verbose=True)

        del fm_backbone, fm_model, fm_trainer

    # ----- Residual model -----
    _print_stage("Preparing residual model")
    residual_type = cfg.adjoint_matching.residual.type
    residual_args = cfg.adjoint_matching.residual.args
    residual_model = get_residual(residual_type, data, residual_args)
    if isinstance(residual_model, torch.nn.Module):
        residual_model = residual_model.to(device)

    use_ema = cfg.adjoint_matching.use_ema_weights

    # ----- Inverse pretraining -----
    if do_inverse_pre:
        _print_stage("Stage 2: Inverse module pretraining")
        if os.path.isfile(inverse_path):
            raise ValueError("File already exists, will not start training.")

        backbone_inverse_type = cfg.adjoint_matching.inverse_module.backbone.type
        backbone_inverse_args = cfg.adjoint_matching.inverse_module.backbone.args
        backbone_inverse = get_backbone(backbone_inverse_type, backbone_inverse_args).to(device)

        fm_backbone_type = cfg.flow_matching.backbone.type
        fm_backbone_args = cfg.flow_matching.backbone.args
        fm_backbone = get_backbone(fm_backbone_type, fm_backbone_args)
        fm_backbone = load_model(fm_backbone, fm_path, use_ema=use_ema)
        fm_model = FlowMatchingModel(fm_backbone, device, **data_args)

        pretrain_inverse(
            fm_model,
            backbone_inverse,
            residual_model,
            inverse_folder,
            prefix_inverse + "_inverse",
            cfg,
            **cfg.adjoint_matching.inverse_pretraining,
        )

        del backbone_inverse, fm_model, fm_backbone

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

        # assemble fine-tuning backbone
        backbone_finetune_type = cfg.adjoint_matching.model_finetune.backbone.type
        backbone_finetune_args = cfg.adjoint_matching.model_finetune.backbone.args
        backbone_finetune = get_backbone_finetune(
            backbone_finetune_type, finetune_base_model, backbone_finetune_args
        )

        am_model = JointAdjointMatchingModel(
            fm_model,
            backbone_finetune,
            backbone_inverse,
            device,
        )

        am_trainer = JointAdjointMatchingTrainer(
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
