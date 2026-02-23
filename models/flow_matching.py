import torch
import torch.nn as nn
import math

from models.image_models.vae import VAE

class FlowMatchingModel(nn.Module):
    def __init__(self,
                 backbone,
                 device,
                 size_data=None,
                 channels_data=1,
                 latent_fm=False):
        super().__init__()

        if size_data is None:
            self.size_data = [64, 64]
        else:
            self.size_data = size_data

        self.channels_data = channels_data
        self.device = device
        self.backbone = backbone.to(device).eval()

        self.latent_fm = latent_fm
        if latent_fm:
            self.vae = self.load_vae()
        else:
            self.vae = None


    def load_vae(self):
        vae = VAE(device=self.device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae

    @staticmethod
    def sigma_memoryless(t, eps=0.01, kappa=0.0):
        eta = (1 - t + eps) / (t + eps)
        sigma = torch.sqrt(2 * (1.-kappa) * eta)
        return sigma

    @staticmethod
    def eta(t, eps=0.01):
        return (1 - t + eps) / (t + eps)


    def sample_noise(self, n_samples):
        return torch.randn([n_samples, self.channels_data] + self.size_data).to(self.device)


    def forward(self, x, t):
        out = self.backbone(x, t)
        return out


    @torch.no_grad()
    def sample_rollout(self, x0=None, n_samples=None, steps=100):

        if x0 is None:
            if n_samples is None:
                raise ValueError('if x0 is not given, n_samples must be specified.')
            x0 = self.sample_noise(n_samples)

        h = 1 / steps
        x_traj = [x0]
        t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        # simple Euler steps along t_steps
        for i in range(steps):
            xt = x_traj[i]
            t = t_steps[i]
            v_t = self.forward(xt, t)
            xt = xt + h * v_t
            x_traj.append(xt)

        x_traj = torch.stack(x_traj, dim=1)
        return x_traj


    @torch.no_grad()
    def sample_memoryless_rollout(self, x0=None, n_samples=None, steps=100):

        if x0 is None:
            if n_samples is None:
                raise ValueError('one of x0 and batch_size needs to be specified.')
            x0 = self.sample_noise(n_samples)

        x_traj = [x0.detach().clone()]

        t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        sigmas = self.sigma_memoryless(t_steps)
        etas = self.eta(t_steps)

        n_steps = len(t_steps) - 1
        for i in range(0, n_steps):
            xt = x_traj[i].detach()
            t = t_steps[i]
            h = t_steps[i + 1] - t_steps[i]

            v_t = self.forward(xt, t)

            if i == (n_steps - 1):
                xt = xt + h * v_t
            else:
                sigma = sigmas[i]
                eta = etas[i]

                noise_x = math.sqrt(h) * sigma * self.sample_noise(xt.shape[0])
                bt = v_t + (sigma ** 2 / (2 * eta)) * (v_t - 1 / (t + h) * xt)
                xt = xt + h * bt + noise_x

            x_traj.append(xt)

        x_traj = torch.stack(x_traj, dim=1)
        return x_traj

    @torch.no_grad()
    def compute_ECI_evolution_BC(self, x0, steps=100, M=5, A_bc_top=0.0, A_bc_bottom=0.0):
        """
        improper, hard-coded implementation of the ECI step into sampling.
        Only works for adjusting the top boundary to 0
        """
        x_traj = [x0]
        t_steps = torch.linspace(0, 1, steps+1).to(self.device)

        nx = x0.shape[-1]
        x = torch.linspace(0.0, 1.0, nx, device=x0.device)

        u_top_y = A_bc_top * torch.sin(torch.pi * x)
        u_bot_y = -A_bc_bottom * torch.sin(torch.pi * x)

        for i in range(steps):
            xt = x_traj[i]
            t = t_steps[i]
            t_next = t_steps[i+1]

            for j in range(M):
                v_t = self.forward(xt, t)
                x_os = xt + (1 - t) * v_t

                x_corr = x_os
                x_corr[:, 0, -1, :] = 0.0
                x_corr[:, 1, -1, :] = u_bot_y
                x_corr[:, 0, 0, :] = 0.0
                x_corr[:, 1, 0, :] = u_top_y

                noise = self.sample_noise(x0.shape[0])
                xt = (1 - t) * noise + t * x_corr

            v_t = self.forward(xt, t)
            x_os = xt + (1 - t) * v_t

            x_corr = x_os
            x_corr[:, 0, -1, :] = 0.0
            x_corr[:, 1, -1, :] = u_bot_y
            x_corr[:, 0, 0, :] = 0.0
            x_corr[:, 1, 0, :] = u_top_y

            noise = self.sample_noise(x0.shape[0])
            xt = (1 - t_next) * noise + t_next * x_corr

            x_traj.append(xt)

        return x_traj



    def generate_pretrain_data(self, n_data, batch_size, steps):
        n_rollouts = n_data // batch_size
        outs = []
        for _ in range(n_rollouts):
            x1 = self.sample_rollout(n_samples=batch_size, steps=steps)[:, -1]

            # if self.latent_fm:
            #     x1 = self.vae(x1)

            outs.append(x1.detach().cpu())
        return torch.cat(outs, dim=0)



