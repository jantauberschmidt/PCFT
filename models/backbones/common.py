import math, torch
from typing import Sequence


def get_spatial_embedding_nd(
        sizes: Sequence[int],          # e.g. (H,W) or (D,H,W)
        d_emb: int = 64,
        device=None
) -> torch.Tensor:
    """
    Return tensor of shape (1, d_model, *sizes)
    containing interleaved [sin, cos] pairs for every spatial axis.

    Requirement: d_emb % (2 * n_dims) == 0
    """
    n_dims = len(sizes)
    assert d_emb % (2 * n_dims) == 0, "d_emb must be multiple of 2*n_dims"
    freqs = d_emb // (2 * n_dims)          # #frequency bands per axis

    # denominator: 10000^{2k/d_model} in classic transformer style
    div_term = torch.exp(
        torch.arange(0, freqs, dtype=torch.float32) *
        -(math.log(10000.0) / freqs)
    ).to(device)

    # create coordinate meshgrid, one per axis
    grids = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32, device=device) for s in sizes],
        indexing="ij"
    )  # each grid_i: shape *sizes

    emb = torch.zeros(d_emb, *sizes, device=device)

    ch = 0
    for f in range(freqs):
        for axis, pos in enumerate(grids):
            emb[ch]   = torch.sin(pos * div_term[f])  # sin channel
            emb[ch+1] = torch.cos(pos * div_term[f])  # cos channel
            ch += 2

    return emb.unsqueeze(0)   # (1, d_model, *sizes)


def get_time_embedding(t: torch.Tensor, d_emb: int = 64) -> torch.Tensor:
    """
    t: (batch, 1) or (batch,) scalar times in [0,1]
    return: (batch, d_model) sinusoidal embedding
    """
    t = t.view(-1, 1)
    div_term = torch.exp(
        torch.arange(0, d_emb, 2, dtype=torch.float32, device=t.device) *
        -(math.log(10000.0) / d_emb)
    )
    emb = torch.zeros(t.size(0), d_emb, device=t.device)
    emb[:, 0::2] = torch.sin(t * div_term)  # even
    emb[:, 1::2] = torch.cos(t * div_term)  # odd
    return emb


def prepare_fno_input_nd(
        u: torch.Tensor,    # (B, C_in, *sizes)
        t: torch.Tensor,    # (B, 1)
        d_emb_space: int = 32,
        d_emb_time: int = 32,
        mode_space='fourier'
) -> torch.Tensor:

    batch, _, *sizes = u.shape
    device = u.device

    # spatial embedding : (1, d_space, *sizes) -> repeat for batch
    if mode_space is None:
        spatial = get_pos_encoding(batch, sizes).to(device)
    else:
        spatial = get_spatial_embedding_nd(sizes, d_emb_space, device).repeat(batch, 1, *[1] * len(sizes))

    # time embedding : (B, d_time) -> reshape & broadcast to spatial dims
    if d_emb_time > 0:
        time = get_time_embedding(t, d_emb_time) # (B,d_time,1)
        for _ in sizes:
            time = time.unsqueeze(-1)            # (B,d_time,1,...)
        time = time.expand(batch, d_emb_time, *sizes)  # (B,d_time,*sizes)

        return torch.cat([u, spatial, time], dim=1)      # (B, C_in+d_sp+d_t, *sizes)
    else:
        return torch.cat([u, spatial], dim=1)      # (B, C_in+d_sp, *sizes)


def get_pos_encoding(batch, sizes):
    axes = [torch.linspace(0., 1., s) for s in sizes]
    mesh = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=0)

    return  mesh.expand(batch, -1, *sizes)
