"""
PickScore-based image quality reward/residual module.

Mathematical specification
--------------------------
For image tensor ``x`` and prompt text ``p``, this module computes

``score(x,p) = 100 * <z_img(x), z_txt(p)>``,

where ``z_img`` and ``z_txt`` are L2-normalized image/text embeddings from the
loaded PickScore model stack. The returned training quantity is the negative:

``residual_or_loss = -score``.

Hence:
- maximizing PickScore corresponds to minimizing returned residual,
- larger returned values indicate worse preference score.

Inputs/Outputs
--------------
- Images: ``(B,3,H,W)`` (or ``(3,H,W)`` auto-batched), float or uint-like.
- `forward` returns raw score ``(B,)``.
- `compute_score` and `compute_residual` return negative score ``(B,)``.
- In `mode='joint'`, `compute_residual` scores `apply_params(img, params)`;
  otherwise it scores raw images and adds ``0.0 * params.squeeze()`` to keep
  graph connectivity to params without changing values.

Gradient, determinism, and failures
-----------------------------------
- Image path is differentiable (no `no_grad` around image encoding).
- Text encoding runs under `torch.no_grad()`; gradients do not flow into prompt
  embeddings or tokenizer/model text tower for optimization.
- Model parameters are frozen (`requires_grad_(False)`), but gradients can flow
  to input pixels and transformation parameters.
- Determinism depends on backend kernels/model ops; this module does not set RNG
  seeds internally.
- External calls to `AutoModel.from_pretrained` / `AutoTokenizer.from_pretrained`
  may require available weights/network/cache; load failures are not caught and
  propagate as exceptions.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer

from models.image_models.color_correction import AugmentPoly

class PickScoreScorer(nn.Module):
    """
    Differentiable preference scorer (PickScore v1).
    Score = 100 * cos(encode_image(x), encode_text(p)).
    - Weights streamed from HF: yuvalkirstain/PickScore_v1
    - Text tokenizer from HF:  laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    - Pure-tensor image preprocessing (no PIL), so autograd works.
    """
    def __init__(self, data, d_poly: int = 3, mode='joint', prompt=None):
        """Initialize scorer, preprocessing pipeline, and prompt configuration."""
        super().__init__()

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.scale = 100.0

        self.mode = mode

        # 1) Load PickScore model (CLIP-H/14 head) and put on device
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)           # usually frozen for reward usage

        # 2) Tokenizer (matches CLIP-H/14 text tower)
        self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

        # 3) Pure-tensor CLIP normalization & geometry (matches CLIP-H)
        image_size = 224
        self.preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(image_size),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        if prompt is None:
            self.prompt = 'a high-quality close-up image of a macaw parrot'
        else:
            self.prompt = prompt
        self.d_poly = int(d_poly)
        self.augment_poly = AugmentPoly(self.d_poly, include_bias=True)

        self.eval()

    @torch.no_grad()
    def encode_text(self, prompts: list[str] | str) -> torch.Tensor:
        """Tokenize prompt(s), encode text features, and L2-normalize embeddings."""
        if isinstance(prompts, str):
            prompts = [prompts]
        tok = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        # AutoModel exposes CLIP-style text features via get_text_features
        t = self.model.get_text_features(**tok).float()  # [B,D]
        return t / t.norm(dim=1, keepdim=True)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode image(s) into normalized embedding vectors.

        Parameters
        ----------
        pixels : torch.Tensor
            Shape ``(B,3,H,W)`` or ``(3,H,W)``; uint8 or float in ``[0,1]`` or
            ``[0,255]``.

        Returns
        -------
        torch.Tensor
            Normalized image embeddings of shape ``(B,D)``.

        Notes
        -----
        This path is differentiable w.r.t. input pixels.
        """
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        x = pixels
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if x.max() > 1:
            x = x / 255.0
        x = self.preprocess(x)                      # keeps autograd graph
        # AutoModel exposes CLIP-style image features via get_image_features
        z = self.model.get_image_features(pixel_values=x).float()  # [B,D]
        return z / z.norm(dim=1, keepdim=True)

    def forward(self, pixels: torch.Tensor, prompts: list[str] | str) -> torch.Tensor:
        """Return PickScore logits (scaled cosine similarity) for each sample."""
        img = self.encode_image(pixels)                           # [B,D] (grad flows to pixels)
        with torch.no_grad():                                     # no grad through text by design
            txt = self.encode_text(prompts)                       # [B,D] or [1,D]
        if txt.shape[0] == 1 and img.shape[0] > 1:
            txt = txt.expand(img.shape[0], -1)
        assert txt.shape[0] == img.shape[0], "batch size mismatch"
        return self.scale * (img * txt).sum(dim=1)

    def compute_score(self, imgs):
        """Return negative PickScore values for ``imgs`` under the stored prompt."""
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)

        scores = self.forward(imgs, self.prompt)
        return -scores


    def compute_residual(self, imgs_raw, params, pretrain=False):
        """Return negative score used as residual/reward objective in training."""
        if imgs_raw.ndim == 3:
            imgs_raw = imgs_raw.unsqueeze(0)

        if self.mode == 'joint':
            imgs = self.apply_params(imgs_raw, params)
            scores = self.forward(imgs, self.prompt)
        else:
            scores = self.forward(imgs_raw, self.prompt) + 0.0 * params.squeeze()

        return -scores

    def apply_params(self, img: torch.Tensor, params: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply polynomial color correction parameters in logit space.

        The transform is:
        1. build polynomial basis via ``AugmentPoly``,
        2. produce additive residual ``res`` by batched matrix multiplication,
        3. map image to logits with clamp+logit, add ``res``, map back with
           sigmoid.
        """

        B, _, H, W = img.shape

        img_ext = self.augment_poly(img)

        res_flat = torch.bmm(params, img_ext.view(B, self.augment_poly.M, H * W))
        res = res_flat.view(B, 3, H, W)

        x_safe = img.clamp(eps, 1.0 - eps)
        logits = torch.logit(x_safe)
        img_mod = torch.sigmoid(logits + res)

        return img_mod


if __name__ == '__main__':
    device = 'mps'
    scorer = PickScoreScorer()
    # prompt = ['test', 'random image noise']
    # imgs = torch.randn(2,3,256,256).to(device)
    # scores = scorer(imgs, prompt)  # [B]; differentiable w.r.t. imgs
    # loss = -scores.mean()
    # loss.backward()
