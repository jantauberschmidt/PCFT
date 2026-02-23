import os

from typing import Any, Dict

import torch

from utils.util import load_model



from models.flow_matching import FlowMatchingModel
from models.joint_adjoint_matching import JointAdjointMatchingModel
from models.adjoint_matching import AdjointMatchingModel

from models.backbones.backbone_dummy import BackboneDummyFinetune, BackboneDummyInverse
from models.backbones.unet_2d import UNet2d
from models.backbones.unet_2d_finetune import UNet2dFinetune
from models.backbones.ufno_2d import UFNO2d
from models.backbones.ufno_2d_finetune import UFNO2dFinetune
from models.image_models.color_correction import PolynomialColorCorrection
from models.image_models.DiT import DiT

from data.darcy_dataset import DarcyDataset
from data.elasticity_dataset import ElasticityDataset
from data.helmholtz_dataset import HelmholtzDataset
from data.stokes_dataset import StokesDataset

from residuals.darcy_residual import WeakDarcyResidual
from residuals.elasticity_residual import WeakElasticityResidual
from residuals.helmholtz_residual import WeakHelmholtzResidual
from residuals.stokes_residual import WeakStokesResidual
from residuals.pick_scorer import PickScoreScorer

_DATASETS: Dict[str, Any] = {
    "darcy": DarcyDataset,
    "elasticity": ElasticityDataset,
    "helmholtz": HelmholtzDataset,
    "stokes": StokesDataset,
    None: None,
}

_BACKBONES: Dict[str, Any] = {
    "DiT": DiT,
    "UNet2d": UNet2d,
    "UNet2dFinetune": UNet2dFinetune,
    "UFNO2d": UFNO2d,
    "UFNO2dFinetune": UFNO2dFinetune,
    "color": PolynomialColorCorrection,
    'DummyFinetune': BackboneDummyFinetune,
    'DummyInverse': BackboneDummyInverse,
}

_RESIDUALS: Dict[str, Any] = {
    "darcy": WeakDarcyResidual,
    "elasticity": WeakElasticityResidual,
    "helmholtz": WeakHelmholtzResidual,
    "stokes": WeakStokesResidual,
    "pickscore": PickScoreScorer,
}


def get_dataset(name: str, path):
    if name is not None:
        return _DATASETS[name](path)
    else:
        return None


def get_backbone(name: str, args):
    return _BACKBONES[name](**args).eval()


def get_backbone_finetune(name: str, base_model, args):
    return _BACKBONES[name](base_model, **args).eval()


def get_residual(name: str, data, args):
    return _RESIDUALS[name](data, **args)


def load_joint_am_model(root_path, rel_path, device, data_paths, load_finetune=True):
    checkpoint_path = os.path.join(root_path, rel_path)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    cfg = ckpt['config']

    prefix_fm = cfg.prefix_fm
    prefix_inverse = cfg.prefix_inverse
    prefix_am = cfg.prefix_am

    fm_folder = os.path.join(root_path, "flow_matching")
    fm_path = os.path.join(fm_folder, prefix_fm + "_fm.pt")

    inverse_folder = os.path.join(root_path, "inverse")
    inverse_path = os.path.join(inverse_folder, prefix_inverse + "_inverse.pt")

    am_folder = os.path.join(root_path, "adjoint_matching")
    am_path = os.path.join(am_folder, prefix_am + "_am.pt")

    data = get_dataset(cfg.dataset, data_paths[cfg.dataset])

    data_args = cfg.data_args
    use_ema = cfg.adjoint_matching.use_ema_weights

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

    if load_finetune:
        backbone_finetune = load_model(backbone_finetune, am_path)

    am_model = JointAdjointMatchingModel(
        fm_model,
        backbone_finetune,
        backbone_inverse,
        device,
    )

    residual_type = cfg.adjoint_matching.residual.type
    residual_args = cfg.adjoint_matching.residual.args
    residual_model = get_residual(residual_type, data, residual_args)
    if isinstance(residual_model, torch.nn.Module):
        residual_model = residual_model.to(device)

    return am_model, data, residual_model, cfg


def load_single_am_model(root_path, rel_path, device, data_paths, frozen_inverse=True):
    checkpoint_path = os.path.join(root_path, rel_path)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    cfg = ckpt['config']

    prefix_fm = cfg.prefix_fm
    prefix_inverse = cfg.prefix_inverse
    prefix_am = cfg.prefix_am

    fm_folder = os.path.join(root_path, "flow_matching")
    fm_path = os.path.join(fm_folder, prefix_fm + "_fm.pt")

    inverse_folder = os.path.join(root_path, "inverse")
    inverse_path = os.path.join(inverse_folder, prefix_inverse + "_inverse.pt")
    inverse_path_ft = os.path.join(inverse_folder, prefix_inverse + "_ft_inverse.pt")


    am_folder = os.path.join(root_path, "adjoint_matching")
    am_path = os.path.join(am_folder, prefix_am + "_am.pt")

    data = get_dataset(cfg.dataset, data_paths[cfg.dataset])

    data_args = cfg.data_args
    use_ema = cfg.adjoint_matching.use_ema_weights

    fm_backbone_type = cfg.flow_matching.backbone.type
    fm_backbone_args = cfg.flow_matching.backbone.args
    fm_backbone = get_backbone(fm_backbone_type, fm_backbone_args)
    fm_backbone = load_model(fm_backbone, fm_path, use_ema=use_ema)
    fm_model = FlowMatchingModel(fm_backbone, device, **data_args)

    # inverse backbone (pretrained)
    backbone_inverse_type = cfg.adjoint_matching.inverse_module.backbone.type
    backbone_inverse_args = cfg.adjoint_matching.inverse_module.backbone.args
    backbone_inverse = get_backbone(backbone_inverse_type, backbone_inverse_args)
    backbone_inverse_base = get_backbone(backbone_inverse_type, backbone_inverse_args)

    backbone_inverse_base = load_model(backbone_inverse, inverse_path)

    if frozen_inverse:
        backbone_inverse = load_model(backbone_inverse, inverse_path)
    else:
        backbone_inverse = load_model(backbone_inverse, inverse_path_ft)


    # base FM again (as finetune base)
    finetune_base_model = get_backbone(fm_backbone_type, fm_backbone_args)
    finetune_base_model = load_model(finetune_base_model, am_path, use_ema=False)


    am_model = AdjointMatchingModel(
        fm_model,
        finetune_base_model,
        backbone_inverse,
        device,
    )

    residual_type = cfg.adjoint_matching.residual.type
    residual_args = cfg.adjoint_matching.residual.args
    residual_model = get_residual(residual_type, data, residual_args)
    if isinstance(residual_model, torch.nn.Module):
        residual_model = residual_model.to(device)

    return am_model, backbone_inverse_base, data, residual_model, cfg