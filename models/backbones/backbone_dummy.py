import torch
import torch.nn as nn


class BackboneDummyFinetune(nn.Module):

    def __init__(self,
                 model_base,
    ):
        super().__init__()

        self.model_base = model_base

        self.param = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    def forward(self, x_in, alpha, vt_alpha_base, t):
        """
        """

        vt_x = self.model_base(x_in, t)

        vt_alpha = torch.zeros_like(vt_alpha_base)

        return vt_x, vt_alpha


class BackboneDummyInverse(nn.Module):

    def __init__(self, output_size=None, output_channels=1
    ):
        super().__init__()

        self.param = nn.Parameter(torch.tensor(0.0))

        if output_size is None:
            self.output_size = [1]
        else:
            self.output_size = output_size

        self.output_channels = output_channels

    def forward(self, x):
        return torch.zeros([x.shape[0], self.output_channels] + self.output_size, device=x.device).requires_grad_(True)