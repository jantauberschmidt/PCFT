import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.channel_mlp import ChannelMLP_nD


# ---------- embeddings & FiLM ----------
class TimeEmbedding(nn.Module):
    """Gaussian Fourier features + MLP → emb_dim."""
    def __init__(self, emb_dim=256, fourier_dim=64):
        super().__init__()
        self.B = nn.Parameter(torch.randn(fourier_dim) * 2.0, requires_grad=False)
        in_dim = 2 * fourier_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):  # t in [0,1], shape [B] or [B,1]
        t = t.view(-1, 1)
        angles = 2 * math.pi * t * self.B.view(1, -1)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(fourier)  # [B, emb_dim]


class AdaGN(nn.Module):
    """Adaptive GroupNorm: scale/shift from embedding."""
    def __init__(self, num_channels, emb_dim, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=1e-6)
        self.to_scale = nn.Linear(emb_dim, num_channels)
        self.to_shift = nn.Linear(emb_dim, num_channels)

    def forward(self, x, emb):
        x = self.gn(x)
        scale = self.to_scale(emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift


# ---------- attention ----------
class SelfAttention2d(nn.Module):
    """Lightweight MHSA for 2D feature maps."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.h = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        # [B, h, C/h, HW]
        def reshape(t):
            t = t.view(B, self.h, C // self.h, H * W)
            return t
        q, k, v = map(reshape, (q, k, v))
        attn = torch.softmax((q.transpose(2,3) @ k) / math.sqrt(C // self.h), dim=-1)  # [B,h,HW,HW]
        out = (attn @ v.transpose(2,3)).transpose(2,3)                                  # [B,h,C/h,HW]
        out = out.contiguous().view(B, C, H, W)
        return self.proj(out) + x  # residual


# ---------- building blocks ----------
class ResBlock(nn.Module):
    """Conv-AdaGN-SiLU-Conv-AdaGN-SiLU with skip proj if needed."""
    def __init__(self, in_ch, out_ch, emb_dim, k=3, groups=32, attn=False, heads=4):
        super().__init__()
        p = (k - 1) // 2
        self.in_ch, self.out_ch = in_ch, out_ch
        self.ada1 = AdaGN(in_ch, emb_dim, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.ada2 = AdaGN(out_ch, emb_dim, groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.attn = SelfAttention2d(out_ch, num_heads=heads) if attn else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.ada1(x, emb)))
        h = self.conv2(F.silu(self.ada2(h, emb)))
        h = h + self.skip(x)
        h = self.attn(h)
        return h


class Downsample2d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(ch_in, ch_out, 1, bias=False)
    def forward(self, x): return self.conv(self.pool(x))


class UpSampleBlock(nn.Module):
    def __init__(self, cin, cout, k=3, mode='bilinear'):
        super().__init__()
        p = (k - 1) // 2
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv = nn.Conv2d(cin, cout, k, padding=p, bias=False)
    def forward(self, x): return F.silu(self.conv(self.up(x)))


class UNet2d(nn.Module):
    """
    4-stage 2D Unet:
      - Pyramid: e.g. 64→32→16→8 (two ResBlocks per stage)
      - FiLM (AdaGN) time conditioning in every block
      - MHSA at low resolutions only (16, 8)
      - Lift/project via ChannelMLP_nD, final concat with lifted input

    Default widths/emb_dim calibrated to ~100M params (± a bit depending on ChannelMLP internals).
    """
    def __init__(
        self,
        in_ch: int = 4,             # typical VAE latent channels
        out_ch: int = 4,
        base=(256, 384, 576, 736), # ~100M with these widths + emb_dim=160 and hidden_lift=512:
        emb_dim: int = 160,
        hidden_lift: int = 512,
        attn_at=(2, 3),  # index of layer (0=top, 3=bottleneck)
        heads: int = 8,
        k: int = 3,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(emb_dim=emb_dim, fourier_dim=64)
        self.lift = ChannelMLP_nD(in_ch, hidden_lift, base[0])

        self.in_channels = in_ch
        self.out_channels = out_ch

        # ---------- legacy attn parametrisation:
        if attn_at:
            if max(attn_at) > 3:
                attn_mapping64 = {64:0,32:1,16:2,8:3}
                attn_at = [attn_mapping64[k] for k in attn_at]

        # ---------- Encoder (64,32,16,8) ----------
        # 64
        self.eb0_0 = ResBlock(base[0], base[0], emb_dim, k, attn=(0 in attn_at), heads=heads)
        self.eb0_1 = ResBlock(base[0], base[0], emb_dim, k, attn=(0 in attn_at), heads=heads)

        self.down1  = Downsample2d(base[0], base[1])  # 64→32
        # 32
        self.eb1_0  = ResBlock(base[1], base[1], emb_dim, k, attn=(1 in attn_at), heads=heads)
        self.eb1_1  = ResBlock(base[1], base[1], emb_dim, k, attn=(1 in attn_at), heads=heads)

        self.down2  = Downsample2d(base[1], base[2])  # 32→16
        # 16  (attention)
        self.eb2_0  = ResBlock(base[2], base[2], emb_dim, k, attn=(2 in attn_at), heads=heads)
        self.eb2_1  = ResBlock(base[2], base[2], emb_dim, k, attn=(2 in attn_at), heads=heads)

        self.down3  = Downsample2d(base[2], base[3])  # 16→8
        # 8   (attention)
        self.eb3_0  = ResBlock(base[3], base[3], emb_dim, k, attn=(3 in attn_at), heads=heads)
        self.eb3_1  = ResBlock(base[3], base[3], emb_dim, k, attn=(3 in attn_at), heads=heads)

        # Bottleneck (8×8, attention on)
        self.bottleneck = ResBlock(base[3], base[3], emb_dim, k, attn=True, heads=heads)

        # ---------- Decoder ----------
        self.up3   = UpSampleBlock(base[3], base[2], k)                 # 8→16
        self.db2_0 = ResBlock(base[2]*2, base[2], emb_dim, k, attn=(2 in attn_at), heads=heads)
        self.db2_1 = ResBlock(base[2],    base[2], emb_dim, k, attn=(2 in attn_at), heads=heads)

        self.up2   = UpSampleBlock(base[2], base[1], k)                 # 16→32
        self.db1_0 = ResBlock(base[1]*2, base[1], emb_dim, k, attn=(1 in attn_at), heads=heads)
        self.db1_1 = ResBlock(base[1],    base[1], emb_dim, k, attn=(1 in attn_at), heads=heads)

        self.up1   = UpSampleBlock(base[1], base[0], k)                 # 32→64
        self.db0_0 = ResBlock(base[0]*2, base[0], emb_dim, k, attn=(0 in attn_at), heads=heads)
        self.db0_1 = ResBlock(base[0],    base[0], emb_dim, k, attn=(0 in attn_at), heads=heads)

        # Final head (concat lifted input)
        self.out_conv = nn.Conv2d(base[0]*2, base[0], k, padding=(k - 1) // 2, bias=True)
        self.project  = ChannelMLP_nD(base[0], hidden_lift, out_ch)

    def forward(self, x, t=None):

        if t is None:
            t = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)

        emb = self.time_emb(t)
        x0  = self.lift(x)

        # enc
        e0 = self.eb0_1(self.eb0_0(x0, emb), emb)              # 64
        e1 = self.eb1_1(self.eb1_0(self.down1(e0), emb), emb)  # 32
        e2 = self.eb2_1(self.eb2_0(self.down2(e1), emb), emb)  # 16
        e3 = self.eb3_1(self.eb3_0(self.down3(e2), emb), emb)  # 8

        b  = self.bottleneck(e3, emb)

        # dec
        d2 = self.up3(b)                                       # 16
        d2 = self.db2_1(self.db2_0(torch.cat([d2, e2], 1), emb), emb)

        d1 = self.up2(d2)                                      # 32
        d1 = self.db1_1(self.db1_0(torch.cat([d1, e1], 1), emb), emb)

        d0 = self.up1(d1)                                      # 64
        d0 = self.db0_1(self.db0_0(torch.cat([d0, e0], 1), emb), emb)

        y  = F.silu(self.out_conv(torch.cat([d0, x0], 1)))
        return self.project(y)



if __name__ == "__main__":
    # standard ~ 100M parameter model
    model = UNet2d(in_ch=4,
                   out_ch=4,
                   base=(256, 384, 576, 736),  # standard configuration for ~100M parameter model
                   emb_dim=160,
                   hidden_lift=512,
                   attn_at=(2,3),
                   heads=8,
                   k=3)


    # ~ 34M parameter model
    model = UNet2d(in_ch=4,
                   out_ch=4,
                   base=(192, 256, 320, 384),
                   emb_dim=128,
                   hidden_lift=256,
                   attn_at=(2,3),
                   heads=8,
                   k=3
                   )


    # smaller "correction" model for added capacity: ~ 8.5M
    model = UNet2d(in_ch=4,
                   out_ch=4,
                   base=(96, 128, 160, 192),
                   emb_dim=64,
                   hidden_lift=256,
                   attn_at=(2,3),
                   heads=8,
                   k=3
                   )

    # Unet used in FNO: ~ 4M
    model = UNet2d(in_ch=32,
                   out_ch=32,
                   base=(64, 96, 96, 128),
                   emb_dim=64,
                   hidden_lift=256,
                   attn_at=(2,3),
                   heads=8,
                   k=3
                   )

