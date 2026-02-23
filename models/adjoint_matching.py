import math
import torch
import torch.nn as nn

class AdjointMatchingModel(nn.Module):
    def __init__(self, base_fm_model,
                 backbone_finetune,
                 backbone_inverse,
                 device
                 ):
        super().__init__()

        self.device = device

        self.base_fm_model = base_fm_model.to(device).eval()
        self.backbone_finetune = backbone_finetune.to(device).eval()
        self.backbone_inverse = backbone_inverse.to(device).eval()

    def alpha_pred(self, x1):
        alpha = self.backbone_inverse(x1)
        return alpha

    def vt_x_base(self, xt, t):
        return self.base_fm_model(xt, t)

    def vt_finetune(self, x, t):
        vt_x = self.backbone_finetune(x, t)
        return vt_x


    @torch.no_grad()
    def sample_memoryless_rollout(self, x0=None, batch_size=None, t_steps=None, steps=None,
                                        kappa_memoryless=0.0):

        if x0 is None:
            if batch_size is None:
                raise ValueError('one of x0 and batch_size needs to be specified.')
            x0 = self.base_fm_model.sample_noise(batch_size)

        x_traj = [x0.detach().clone()]
        x_base_traj = [x0.detach().clone()]

        if t_steps is None:
            if steps is None:
                raise ValueError('one of t_steps and steps needs to be specified.')
            t_steps = torch.linspace(0, 1, steps + 1).to(self.device)


        sigmas = self.base_fm_model.sigma_memoryless(t_steps, kappa=kappa_memoryless)
        etas = self.base_fm_model.eta(t_steps)

        n_steps = len(t_steps) - 1
        for i in range(0, n_steps):
            xt = x_traj[i].detach()
            xt_base = x_base_traj[i].detach()
            t = t_steps[i]
            h = t_steps[i+1] - t_steps[i]

            vt_x_base_base = self.vt_x_base(xt_base, t)  # base vector field at base trajectory
            vt_x = self.vt_finetune(xt, t)

            if i == (n_steps-1):
                xt = xt + h * vt_x
                xt_base = xt_base + h * vt_x_base_base
            else:
                sigma = sigmas[i]
                eta = etas[i]

                noise_x = math.sqrt(h) * sigma * self.base_fm_model.sample_noise(xt.shape[0])
                bt = vt_x  + (sigma ** 2 / (2 * eta)) * (vt_x - 1 / (t+h) * xt)
                xt = xt + h * bt + noise_x

                bt_base = vt_x_base_base + (sigma ** 2 / (2 * eta)) * (vt_x_base_base - 1 / (t + h) * xt_base)
                xt_base = xt_base + h * bt_base + noise_x


            x_traj.append(xt)
            x_base_traj.append(xt_base)

        return x_traj, x_base_traj, t_steps


    @torch.no_grad()
    def sample_rollout(self, x0=None, batch_size=None, t_steps=None, steps=None):

        if x0 is None:
            if batch_size is None:
                raise ValueError('one of x0 and batch_size needs to be specified.')
            x0 = self.base_fm_model.sample_noise(batch_size)



        x_traj = [x0.detach().clone()]
        x_base_traj = [x0.detach().clone()]

        if t_steps is None:
            if steps is None:
                raise ValueError('one of t_steps and steps needs to be specified.')
            t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        n_steps = len(t_steps) - 1
        for i in range(0, n_steps):
            xt = x_traj[i].detach()
            xt_base = x_base_traj[i].detach()
            t = t_steps[i]
            h = t_steps[i + 1] - t_steps[i]

            vt_x_base_base = self.vt_x_base(xt_base, t)  # base vector field at base trajectory

            vt_x = self.vt_finetune(xt, t)

            xt = xt + h * vt_x
            xt_base = xt_base + h * vt_x_base_base

            x_traj.append(xt)
            x_base_traj.append(xt_base)

        return x_traj, x_base_traj

    @torch.no_grad()
    def generate_samples(self, n_samples):
        x_traj, _ = self.sample_rollout(batch_size=n_samples, steps=100)
        x1 = x_traj[-1]
        alpha1 = self.alpha_pred(x1)
        return x1, alpha1


