import math
import torch
import torch.nn as nn

class JointAdjointMatchingModel(nn.Module):
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

        self.size_alpha = self.backbone_inverse.output_size
        self.channels_alpha = self.backbone_inverse.output_channels


    def sample_alpha(self, n_samples):
        return torch.randn([n_samples, self.channels_alpha] + self.size_alpha).to(self.device)


    def alpha_pred(self, x1):
        alpha = self.backbone_inverse(x1)
        return alpha


    def vt_x_base(self, xt, t):
        return self.base_fm_model(xt, t)


    def vt_alpha_base(self, xt, alpha_t, vt_x, t):
        # return vector pointing from alpha towards predicted alpha of extrapolated x1
        x1_hat = xt + (1 - t) * vt_x
        return (self.alpha_pred(x1_hat) - alpha_t) / (1 - t).clamp(min=1e-3)


    def vt_finetune(self, x, alpha, vt_alpha_base, t):
        vt_x, vt_alpha = self.backbone_finetune(x, alpha, vt_alpha_base, t)
        return vt_x, vt_alpha


    @torch.no_grad()
    def sample_memoryless_rollout_joint(self, x0=None, alpha_0=None, batch_size=None, t_steps=None, steps=None,
                                        kappa_memoryless=0.0):

        if x0 is None:
            if batch_size is None:
                raise ValueError('one of x0 and batch_size needs to be specified.')
            x0 = self.base_fm_model.sample_noise(batch_size)

        if alpha_0 is None:
            if batch_size is None:
                raise ValueError('one of alpha_0 and batch_size needs to be specified.')
            alpha_0 = self.sample_alpha(batch_size)

        x_traj = [x0.detach().clone()]
        x_base_traj = [x0.detach().clone()]
        alpha_traj = [alpha_0.detach().clone()]

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
            alpha_t = alpha_traj[i]

            vt_x_base_base = self.vt_x_base(xt_base, t)  # base vector field at base trajectory
            vt_x_base = self.vt_x_base(xt, t)  # base vector field at fine-tuned trajectory
            vt_alpha_base = self.vt_alpha_base(xt, alpha_t, vt_x_base, t)

            vt_x, vt_alpha = self.vt_finetune(xt, alpha_t, vt_alpha_base, t)

            if i == (n_steps-1):
                xt = xt + h * vt_x
                xt_base = xt_base + h * vt_x_base_base
                alpha_t = alpha_t + h * vt_alpha
            else:
                sigma = sigmas[i]
                eta = etas[i]

                noise_x = math.sqrt(h) * sigma * self.base_fm_model.sample_noise(xt.shape[0])
                bt = vt_x  + (sigma ** 2 / (2 * eta)) * (vt_x - 1 / (t+h) * xt)
                xt = xt + h * bt + noise_x

                bt_base = vt_x_base_base + (sigma ** 2 / (2 * eta)) * (vt_x_base_base - 1 / (t + h) * xt_base)
                xt_base = xt_base + h * bt_base + noise_x

                noise_alpha = math.sqrt(h) * sigma * self.sample_alpha(alpha_t.shape[0])
                bt_alpha = vt_alpha  + (sigma ** 2 / (2 * eta)) * (vt_alpha - 1 / (t+h) * alpha_t)
                alpha_t = alpha_t + h * bt_alpha + noise_alpha


            x_traj.append(xt)
            x_base_traj.append(xt_base)
            alpha_traj.append(alpha_t)

        return x_traj, x_base_traj, alpha_traj, t_steps


    @torch.no_grad()
    def sample_rollout_joint(self, x0=None, alpha_0=None, batch_size=None, t_steps=None, steps=None):

        if x0 is None:
            if batch_size is None:
                raise ValueError('one of x0 and batch_size needs to be specified.')
            x0 = self.base_fm_model.sample_noise(batch_size)

        if alpha_0 is None:
            if batch_size is None:
                raise ValueError('one of alpha_0 and batch_size needs to be specified.')
            alpha_0 = self.sample_alpha(batch_size)


        x_traj = [x0.detach().clone()]
        x_base_traj = [x0.detach().clone()]
        alpha_traj = [alpha_0.detach().clone()]

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
            alpha_t = alpha_traj[i]

            vt_x_base_base = self.vt_x_base(xt_base, t)  # base vector field at base trajectory
            vt_x_base = self.vt_x_base(xt, t)  # base vector field at fine-tuned trajectory
            vt_alpha_base = self.vt_alpha_base(xt, alpha_t, vt_x_base, t)

            vt_x, vt_alpha = self.vt_finetune(xt, alpha_t, vt_alpha_base, t)


            xt = xt + h * vt_x
            xt_base = xt_base + h * vt_x_base_base
            alpha_t = alpha_t + h * vt_alpha


            x_traj.append(xt)
            x_base_traj.append(xt_base)
            alpha_traj.append(alpha_t)

        return x_traj, x_base_traj, alpha_traj

    @torch.no_grad()
    def generate_samples(self, n_samples):
        x_traj, _, alpha_traj = self.sample_rollout_joint(batch_size=n_samples, steps=100)
        x1 = x_traj[-1]
        alpha1 = alpha_traj[-1]
        return x1, alpha1


