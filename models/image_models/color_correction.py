import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentPoly(nn.Module):
    """
    Fixed polynomial channel augmentation for 3-channel BCHW inputs.

    Order matches your previous augment_poly:
      - degree = 1: pure powers [R, G, B]
      - for d = 2..D: pure first [R^d, G^d, B^d], then mixed terms in lexicographic order
        of exponent tuples e = (a,b,c) with a+b+c = d and at least two positive entries.
      - optional bias appended as a final constant-1 channel.

    Returns feats of shape (B, M + bias, H, W) where M = comb(3 + D, D) - 1.
    """

    def __init__(self, degree: int, include_bias: bool = True):
        super().__init__()
        assert isinstance(degree, int) and degree >= 1
        self.degree = degree
        self.include_bias = include_bias

        # Precompute exponent tuples E with the exact same ordering semantics.
        exps = []
        # degree = 1: pure powers R, G, B
        exps.extend([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        # degrees 2..D: pure first, then mixed with lexicographic compositions
        for d in range(2, degree + 1):
            exps.extend([(d, 0, 0), (0, d, 0), (0, 0, d)])  # pure
            for e in self._compositions_lex(d, 3):
                if (e[0] > 0) + (e[1] > 0) + (e[2] > 0) >= 2:
                    exps.append(tuple(e))

        E = torch.tensor(exps, dtype=torch.long)  # (M, 3)
        self.register_buffer("E", E, persistent=False)  # moves with .to(device)

        # Cache sizes
        if self.include_bias:
            self.M = E.shape[0] + 1
        else:
            self.M = E.shape[0]


    @staticmethod
    def _compositions_lex(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for t in range(total + 1):
            for rest in AugmentPoly._compositions_lex(total - t, parts - 1):
                yield (t,) + rest

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, 3, H, W) with gradient.
        returns feats: (B, M + bias, H, W)
        """
        assert img.ndim == 4 and img.size(1) == 3

        # Per-channel powers up to degree D (inclusive).
        # powers[c][k] = x_c^k, for k=0..D, with powers[c][0] = 1.
        R, G, Bc = img[:, 0:1], img[:, 1:2], img[:, 2:3]  # keep shapes (B,1,H,W)
        one = torch.ones_like(R)
        R_pows = [one, R]
        G_pows = [one, G]
        B_pows = [one, Bc]
        for _ in range(2, self.degree + 1):
            R_pows.append(R_pows[-1] * R)
            G_pows.append(G_pows[-1] * G)
            B_pows.append(B_pows[-1] * Bc)

        # Assemble monomials according to precomputed exponents.
        feats_list = []
        # Degree 1 block (first 3 tuples): R,G,B fast path
        feats_list.extend([R_pows[1], G_pows[1], B_pows[1]])
        # Remaining tuples (including degree >=2): multiply selected powers
        for a, b, c in self.E.tolist()[3:]:
            feats_list.append(R_pows[a] * G_pows[b] * B_pows[c])

        feats = torch.cat(feats_list, dim=1)  # (B, M, H, W)
        if self.include_bias:
            feats = torch.cat([feats, one], dim=1)  # append bias
        return feats




class PolynomialColorCorrection(nn.Module):
    """
    Color correction via polynomial parametrisation with per-stream LayerNorm and
    degree-wise LayerNorm + learnable gains on the coefficient output.

    Output coefficients are ordered exactly as in AugmentPoly (degree 1, then 2..D, then bias if present).
    """

    def __init__(self,
                 d_poly: int = 3,
                 d_proj: int = 128,
                 d_hidden: int = 64):
        super().__init__()


        self.d_poly = int(d_poly)
        self.augment_poly = AugmentPoly(self.d_poly, include_bias=True)


        self.output_size = [self.augment_poly.M]
        self.output_channels = 3

        self.stem = nn.Sequential(
            nn.Conv2d(4, d_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_hidden),
            nn.SiLU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(d_hidden, d_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_hidden),
            nn.SiLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(d_hidden, d_hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, d_hidden),
            nn.SiLU()
        )  # 32→16

        self.block2 = nn.Sequential(
            nn.Conv2d(d_hidden, d_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_hidden),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(d_hidden, d_hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, d_hidden),
            nn.SiLU()
        )  # 16→8

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_hidden, max(8, d_hidden // 8), kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(max(8, d_hidden // 8), d_hidden, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.moments_dim = 4 * 4  # per-channel mean, std, skew, kurt for 4 latent channels
        fused_in = d_hidden * 3 + self.moments_dim

        self.p_embed = nn.Linear(fused_in, d_proj, bias=False)

        self.ln = nn.LayerNorm(d_proj, elementwise_affine=True)
        self.ffn1 = nn.Linear(d_proj, 4 * d_proj, bias=True)
        self.ffn2 = nn.Linear(4 * d_proj, d_proj, bias=True)
        nn.init.zeros_(self.ffn2.weight)
        nn.init.zeros_(self.ffn2.bias)

        self.head = nn.Linear(d_proj, 3 * self.augment_poly.M, bias=True)

    @staticmethod
    def _channel_moments(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xf = x.view(b, c, -1)
        mean = xf.mean(dim=-1)
        xc = xf - mean.unsqueeze(-1)
        var = (xc ** 2).mean(dim=-1) + 1e-8
        std = var.sqrt()
        z = xc / std.unsqueeze(-1)
        skew = (z ** 3).mean(dim=-1)
        kurt = (z ** 4).mean(dim=-1)
        return torch.cat([mean, std, skew, kurt], dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.stem(z)

        x = self.block1(x)
        g32 = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        x = self.down1(x)
        x = self.block2(x)
        g16 = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        x = self.down2(x)
        w = self.se(x)
        x = x * w
        g8 = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        mom = self._channel_moments(z)

        feat = torch.cat([g32, g16, g8, mom], dim=1)
        h = self.p_embed(feat)

        h_n = self.ln(h)
        h_n = F.silu(self.ffn1(h_n))
        h = h + self.ffn2(h_n)

        alpha = self.head(h)  # (B, 3*self.augment_poly.M)
        return alpha.view(z.size(0), 3, self.augment_poly.M)

    # implemented here and in aesthetics reward for convenience
    def apply_params(self, img: torch.Tensor, params: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

        B, _, H, W = img.shape

        img_ext = self.augment_poly(img)

        res_flat = torch.bmm(params, img_ext.view(B, self.augment_poly.M, H * W))
        res = res_flat.view(B, 3, H, W)

        x_safe = img.clamp(eps, 1.0 - eps)
        logits = torch.logit(x_safe)
        img_mod = torch.sigmoid(logits + res)

        return img_mod
