import torch
from torch import nn

from diffusers import AutoencoderKL

class VAE(nn.Module):
    """
    AutoencoderKL VAE
    expects inputs to be [B,C,H,W] in [0, 1], internally converts to [-1, 1]
    """
    def __init__(self, device='cpu', vae_model='stabilityai/sd-vae-ft-mse'):
        super().__init__()

        # other option for vae model: "stabilityai/sdxl-vae"
        self._exclude_from_saving = True

        self.model = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch.float32).to(device).eval()
        self.scaling_factor = self.model.config.scaling_factor

        self.eval()

    def encode_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        deterministic: returns mean value of conditional latent distribution
        """
        pixel_values = pixel_values * 2.0 - 1.0
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        latents = self.model.encode(pixel_values).latent_dist.mean
        return latents * self.scaling_factor

    def decode_latents(self, latents_scaled: torch.Tensor) -> torch.Tensor:
        imgs = self.model.decode(latents_scaled / self.scaling_factor).sample
        return (imgs.clamp(-1, 1) + 1) * 0.5

    def forward(self, latents_scaled: torch.Tensor) -> torch.Tensor:
        return self.decode_latents(latents_scaled)