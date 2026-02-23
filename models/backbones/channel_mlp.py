import torch
import torch.nn as nn

class ChannelMLP_nD(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, C, *spatial]
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]                      # arbitrary spatial dims
        x = x.view(B, C, -1)                        # [B, C, S]

        if x.device.type == "mps":
            # maximum supported length on MPS is 65_536 -> chunk it up
            x = chunk_apply(x, self.mlp)
        else:
            x = self.mlp(x)                                   # [B, C', S]

        x = x.view(B, -1, *spatial_shape)                 # [B, C', *spatial]
        return x


def chunk_apply(x: torch.Tensor, fn, chunk_size: int = 65_536) -> torch.Tensor:
    """
    Apply `fn` to `x` in pieces no larger than `chunk_size` along dim -1
    and re‑concatenate the results.
    """
    if x.size(-1) <= chunk_size:          # fast path – no chunking needed
        return fn(x)

    # Split, apply, gather
    chunks = x.split(chunk_size, dim=-1)  # tuple of Tensors
    outs   = [fn(c) for c in chunks]     # gradients propagate normally
    return torch.cat(outs, dim=-1)