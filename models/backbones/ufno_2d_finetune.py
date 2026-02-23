
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.channel_mlp import ChannelMLP_nD
from models.backbones.common import prepare_fno_input_nd
from models.backbones.unet_2d import UNet2d

class UFNO2dFinetune(nn.Module):
    """
    Adds output heads and input fusion for alpha, plus a correction term for x (to add more tunable capacity)
    """

    def __init__(self,
                 ufno_base,
                 # parameters for correction unets
                 base_unet=(96, 128, 160, 192),
                 emb_unet: int = 96,
                 hidden_unet: int = 256,
                 attn_at=(2, 3),
                 heads_unet: int = 8,
                 # alpha unet parameters
                 d_alpha: int = 1,
    ):
        super().__init__()

        self.ufno_base = ufno_base

        in_lift = self.ufno_base.lifting.in_dim
        in_base = self.ufno_base.input_channels

        self.unet_correction = UNet2d(in_ch=ufno_base.width_fno + in_base + d_alpha,
                                      out_ch=ufno_base.projection.out_dim,
                                      base=base_unet,
                                      emb_dim=emb_unet,
                                      hidden_lift=hidden_unet,
                                      attn_at=attn_at,
                                      heads=heads_unet,
                                      k=3)
        # zero-init projection of unet_correction
        nn.init.zeros_(self.unet_correction.project.mlp[-1].weight)
        nn.init.zeros_(self.unet_correction.project.mlp[-1].bias)

        self.pixel_wise_correction = ChannelMLP_nD(ufno_base.width_fno + 2 + in_base,
                                                   hidden_unet,
                                                   ufno_base.projection.out_dim)
        nn.init.zeros_(self.pixel_wise_correction.mlp[-1].weight)
        nn.init.zeros_(self.pixel_wise_correction.mlp[-1].bias)

        self.alpha_projection = UNet2d(in_ch=in_lift + 2 * d_alpha,
                                      out_ch=d_alpha,
                                      base=base_unet,
                                      emb_dim=emb_unet,
                                      hidden_lift=hidden_unet,
                                      attn_at=attn_at,
                                      heads=heads_unet,
                                      k=3)
        # zero-init projection of alpha
        nn.init.zeros_(self.alpha_projection.project.mlp[-1].weight)
        nn.init.zeros_(self.alpha_projection.project.mlp[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, x_in, alpha, vt_alpha_base, t):
        """
        x              : (B, Cin,  H, W)
        alpha          : (B, 1,    H, W)   conditioning field
        vt_alpha_base  : (B, 1,    H, W)   baseline alpha vector‑field
        """

        base = self.ufno_base

        x_pad = prepare_fno_input_nd(x_in, t, base.d_emb_space, base.d_emb_time)
        if base.pad_mode is not None:
            x_pad = F.pad(x_pad, base.pad, mode=base.pad_mode)
            alpha = F.pad(alpha, base.pad, mode=base.pad_mode)
            vt_alpha_base = F.pad(vt_alpha_base, base.pad, mode=base.pad_mode)

        x = base.lifting(x_pad)


        # ------------------------- UFNO core ---------------

        for i in range(base.n_layers):
            x1 = base.spectral[i](x)
            x2 = base.skip[i](x)
            x3 = base.unet[i](x, t)
            x = F.gelu(x1 + x2 + x3)

        vt_x = base.projection(x)
        correction_x = self.unet_correction(torch.cat([x, vt_x, alpha], dim=1), t)
        vt_x = vt_x + correction_x

        # ------------------------- alpha head --------------------------
        correction_alpha = self.alpha_projection(torch.cat([x_pad, alpha, vt_alpha_base], dim=1), t)
        vt_alpha = vt_alpha_base + correction_alpha

        # crop back to original size
        vt_x = vt_x[:, :, base.h:-base.h, base.h:-base.h]
        vt_alpha = vt_alpha[:, :, base.h:-base.h, base.h:-base.h]

        # final pixel-wise correction for quick adaptation
        x = x[:, :, base.h:-base.h, base.h:-base.h]
        x_pos = prepare_fno_input_nd(x, t, base.d_emb_space, 0, mode_space=None)  # only append two positional encoding layers
        pixel_correction_x = self.pixel_wise_correction(torch.cat([x_pos, vt_x], dim=1))
        vt_x = vt_x + pixel_correction_x

        return vt_x, vt_alpha