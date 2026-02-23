import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.unet_2d import UNet2d


class AlphaCorrection(nn.Module):
    """
    Color correction via polynomial parametrisation (degree d_poly) conditioned on:
      - latent z ∈ R^{B,4,32,32}
      - current alpha ∈ R^{B,3,M}
      - vt_alpha_base ∈ R^{B,3,M}

    Conditioning is done by token-level fusion and AdaLN/FiLM on the head,
    plus a mild SE rescaling of conv features. No spatial broadcasting needed.
    """

    def __init__(self,
                 d_alpha: int = 20,
                 d_proj: int = 128,
                 d_hidden: int = 64):
        super().__init__()

        self.M = d_alpha

        # -------- Conv stem (unchanged) --------
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

        # SE gate (we'll modulate it with alpha/vt token later)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_hidden, max(8, d_hidden // 8), kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(max(8, d_hidden // 8), d_hidden, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Global latent statistics
        self.moments_dim = 4 * 4  # per-latent-channel mean, std, skew, kurt (4 stats × 4 chans)

        # Image token from multi-scale pooled conv features (+ moments)
        fused_in = d_hidden * 3 + self.moments_dim
        self.p_img = nn.Linear(fused_in, d_proj, bias=False)

        # ---------- New: alpha / vt projections & fusion ----------
        d_flat = 3 * self.M
        self.p_alpha = nn.Linear(d_flat, d_proj, bias=False)
        self.p_vt    = nn.Linear(d_flat, d_proj, bias=False)

        # Small gated fusion MLP over {img, alpha, vt} tokens with multiplicative interactions
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_proj + d_proj, d_proj, bias=True),  # includes interaction token (see forward)
            nn.GroupNorm(8, d_proj),
            nn.SiLU(),
            nn.Linear(d_proj, d_proj, bias=True),
            nn.SiLU()
        )
        # Produce AdaLN parameters (γ, β) for the head token and a light SE scale
        self.to_gamma_beta = nn.Linear(d_proj, 2 * d_proj, bias=True)
        self.to_se_scale   = nn.Linear(d_proj, d_hidden, bias=True)

        # Head: Pre-LN residual MLP with AdaLN modulation
        self.ln = nn.LayerNorm(d_proj, elementwise_affine=False)  # affine comes from AdaLN
        self.ffn1 = nn.Linear(d_proj, 4 * d_proj, bias=True)
        self.ffn2 = nn.Linear(4 * d_proj, d_proj, bias=True)
        nn.init.zeros_(self.ffn2.weight)
        nn.init.zeros_(self.ffn2.bias)

        # Final coefficients (absolute, not delta)
        self.head = nn.Linear(d_proj, 3 * self.M, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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
        return torch.cat([mean, std, skew, kurt], dim=1)  # (B, 4*C)

    def forward(self, z: torch.Tensor, alpha: torch.Tensor, vt_alpha_base: torch.Tensor) -> torch.Tensor:
        """
        z:             (B, 4, 32, 32)
        alpha:         (B, 3, M)
        vt_alpha_base: (B, 3, M)
        returns: alpha_hat (B, 3, M)
        """
        B = z.size(0)
        # --- Conv path to make image token ---
        x = self.stem(z)
        x = self.block1(x)
        g32 = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.down1(x)
        x = self.block2(x)
        g16 = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.down2(x)
        w = self.se(x)                              # (B, d_hidden, 1, 1)
        # Light conditioning: scale SE with alpha/vt (computed below), applied after we get the token.
        g8  = F.adaptive_avg_pool2d(x * w, 1).squeeze(-1).squeeze(-1)
        mom = self._channel_moments(z)
        img_tok = self.p_img(torch.cat([g32, g16, g8, mom], dim=1))  # (B, d_proj)

        # --- Alpha/vt tokens & fusion ---
        a_tok  = self.p_alpha(alpha.view(B, -1))                     # (B, d_proj)
        vt_tok = self.p_vt(vt_alpha_base.view(B, -1))                # (B, d_proj)

        # Multiplicative interactions help alignment without huge params
        inter = img_tok * a_tok + img_tok * vt_tok + a_tok * vt_tok  # (B, d_proj)
        fused = self.fuse(torch.cat([img_tok, a_tok, vt_tok, inter], dim=-1))  # (B, d_proj)

        # AdaLN params and SE modulation from fused token
        gb = self.to_gamma_beta(fused)                               # (B, 2*d_proj)
        gamma, beta = gb.chunk(2, dim=-1)                            # (B, d_proj), (B, d_proj)
        se_scale = torch.sigmoid(self.to_se_scale(fused)).view(B, -1, 1, 1)  # (B, d_hidden,1,1)

        # Recompute last pooled token with conditioned SE gate (cheap and stable)
        g8_c = F.adaptive_avg_pool2d(x * (w * se_scale), 1).squeeze(-1).squeeze(-1)
        img_tok_c = self.p_img(torch.cat([g32, g16, g8_c, mom], dim=1))
        h = img_tok_c

        # AdaLN on head token
        h_n = self.ln(h)
        h_mod = (1.0 + gamma) * h_n + beta
        h1 = F.silu(self.ffn1(h_mod))
        h = h + self.ffn2(h1)

        coeff = self.head(h)                                        # (B, 3*M)
        return coeff.view(B, 3, self.M)


class UNet2dFinetune(nn.Module):

    def __init__(self,
                 unet_base,
                 # alpha head parameters
                 d_alpha: int = 1,
                 k_alpha: int = 16,
                 d_proj_alpha:int = 128,
                 d_hidden_alpha: int = 64,
                 # unet (x) parameters
                 base_unet = (96, 128, 160, 192),
                 emb_unet: int = 96,
                 hidden_unet: int = 256,
                 attn_at=(2,3),
                 heads_unet: int = 8,
    ):
        super().__init__()

        self.d_alpha = d_alpha
        self.k_alpha = k_alpha


        self.unet_base = unet_base

        self.alpha_map = nn.Sequential(
            nn.Linear(d_alpha * 3, hidden_unet),  # flatten α first
            nn.BatchNorm1d(hidden_unet),
            nn.ReLU(),
            nn.Linear(hidden_unet, k_alpha)
        )

        self.unet_correction = UNet2d(in_ch=2 * unet_base.in_channels + k_alpha,
                                      out_ch=unet_base.out_channels,
                                      base=base_unet,
                                      emb_dim=emb_unet,
                                      hidden_lift=hidden_unet,
                                      attn_at=attn_at,
                                      heads=heads_unet,
                                      k=3)

        # zero-init projection of unet_correction
        nn.init.zeros_(self.unet_correction.project.mlp[-1].weight)
        nn.init.zeros_(self.unet_correction.project.mlp[-1].bias)

        # v_alpha head (residual on top of base flow)
        self.alpha_projection = AlphaCorrection(d_alpha=d_alpha,
                                                d_proj=d_proj_alpha,
                                                d_hidden=d_hidden_alpha)


    # ------------------------------------------------------------------
    def forward(self, x_in, alpha, vt_alpha_base, t):
        """
        x_in:          [B, C, H, W]
        alpha:         [B, 3, D]
        vt_alpha_base: [B, 3, D]
        t:             [B] or [B,1]
        """
        B, C, H, W = x_in.shape

        # base image drift
        vt_x_base = self.unet_base(x_in, t)

        # α token and broadcasted maps for concat to image path
        alpha_token = self.alpha_map(alpha.view(B, -1))  # [B, K]
        alpha_map = alpha_token.view(B, self.k_alpha, 1, 1).expand(B, self.k_alpha, H, W)

        # image correction path (just extra capacity)
        correction_x = self.unet_correction(torch.cat([vt_x_base, x_in, alpha_map], dim=1), t)
        vt_x = vt_x_base + correction_x

        # α evolution head (residual on top of vt_alpha_base)
        delta_alpha = self.alpha_projection(x_in, alpha, vt_alpha_base)
        vt_alpha = vt_alpha_base + delta_alpha

        return vt_x, vt_alpha