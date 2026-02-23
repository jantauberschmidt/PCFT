
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.common import prepare_fno_input_nd
from models.backbones.channel_mlp import ChannelMLP_nD
from models.backbones.unet_2d import UNet2d

class SpectralConv2d(nn.Module):
    """2‑D Fourier layer with truncated complex weights."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2

        scale = 1 / in_channels ** 0.5

        shape = (in_channels, out_channels, modes1, modes2)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(4)
        ])

    @staticmethod
    def _cmm(a, w):
        """complex matrix‑multiplication helper"""
        return torch.einsum("bixy,ioxy->boxy", a, w)

    def forward(self, x):  # x:(B,Cin,H,W)
        B, _, H, W = x.shape

        # rfft2 not implemented for compressed data types, use float and convert back after
        x_ft = torch.fft.rfft2(x.float(), dim=(-2, -1))
        W_ft = W // 2 + 1
        out_ft = torch.zeros(B, self.weights[0].shape[1], H, W_ft,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        a, b, c, d = self.weights
        out_ft[:, :, :m1, :m2] = self._cmm(x_ft[:, :, :m1, :m2], a)
        out_ft[:, :, -m1:, :m2] = self._cmm(x_ft[:, :, -m1:, :m2], b)
        out_ft[:, :, :m1, -m2:] = self._cmm(x_ft[:, :, :m1, -m2:], c)
        out_ft[:, :, -m1:, -m2:] = self._cmm(x_ft[:, :, -m1:, -m2:], d)
        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)



class UFNO2d(nn.Module):
    def __init__(self,
                 n_layers: int = 4,
                 pad_mode: str = "reflect",
                 in_channels: int = 1,
                 d_emb_time: int = 32,
                 d_emb_space: int = 32,
                 out_channels: int = 1,
                 hidden_lift: int = 256,
                 # params FNO
                 modes_fno: tuple[int, int] = (32, 32),
                 width_fno: int = 32,
                 # params Unet
                 base_unet=(64, 96, 96, 128),
                 emb_unet: int = 64,
                 attn_at=(2, 3),
                 heads: int = 8,
                 output_size = None,
                 **kwargs):
        super().__init__()

        assert len(modes_fno) == 2, "2D U-FNO expects number of Fourier modes given for both axes."

        self.pad_mode = pad_mode
        self.h = 2 ** n_layers
        if self.pad_mode is not None:
            self.pad = (self.h, self.h, self.h, self.h)
        else:
            self.pad = None


        self.width_fno = width_fno
        self.n_modes = modes_fno
        self.d_emb_time = d_emb_time
        self.d_emb_space = d_emb_space

        self.input_channels = in_channels

        self.lifting = ChannelMLP_nD(in_channels + d_emb_space + d_emb_time, hidden_lift, width_fno)

        self.n_layers = n_layers

        self.spectral = nn.ModuleList([SpectralConv2d(width_fno,
                                                      width_fno,
                                                      self.n_modes[0],
                                                      self.n_modes[1])
                                       for _ in range(self.n_layers)])

        self.skip = nn.ModuleList([nn.Conv2d(width_fno, width_fno, kernel_size=1) for _ in range(self.n_layers)])


        self.unet = nn.ModuleList([UNet2d(in_ch=width_fno,
                                          out_ch=width_fno,
                                          base=base_unet,
                                          emb_dim=emb_unet,
                                          hidden_lift=hidden_lift,
                                          attn_at=attn_at,
                                          heads=heads,
                                          k=3,
                                          ) for _ in range(self.n_layers)])


        self.projection = ChannelMLP_nD(width_fno, hidden_lift, out_channels)

        self.output_size = output_size
        self.output_channels = out_channels


    def forward(self, x, t=None):  # (B,C_in,H,W)

        if t is None:
            t = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)

        x = prepare_fno_input_nd(x, t, self.d_emb_space, self.d_emb_time)
        if self.pad_mode is not None:
            x = F.pad(x, self.pad, mode=self.pad_mode)

        x = self.lifting(x)

        for i in range(self.n_layers):
            x1 = self.spectral[i](x)
            x2 = self.skip[i](x)
            x3 = self.unet[i](x, t)
            x = F.gelu(x1 + x2 + x3)

        out = self.projection(x)

        if self.pad_mode is not None:
            return out[:, :, self.h:-self.h, self.h:-self.h]
        else:
            return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":

    # ~ 35M parameters
    model = UFNO2d(n_layers=4,
                   pad_mode="reflect",
                   in_channels= 1,
                   d_emb_time= 32,
                   d_emb_space= 32,
                   out_channels= 1,
                   hidden_lift= 256,
                   # params FNO
                   modes_fno=(32, 32),
                   width_fno= 32,
                   # params Unet
                   base_unet=(64, 96, 96, 128),
                   emb_unet=64,
                   attn_at=(2, 3),
                   heads=8)

    model.count_params()
