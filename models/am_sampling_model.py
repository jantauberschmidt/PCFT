import torch
import torch.nn.functional as F
from models.joint_adjoint_matching import JointAdjointMatchingModel



class JointAdjointMatchingSamplingModel(JointAdjointMatchingModel):
    def __init__(self, base_fm_model,
                 backbone_finetune,
                 backbone_inverse,
                 device,
                 data):
        super().__init__(base_fm_model,
                 backbone_finetune,
                 backbone_inverse,
                 device)

        self.data = data

    def denormalize_alpha(self, alpha):
        alpha = self.data.denormalize_alpha(alpha)
        return alpha

    def denormalize_x(self, x):
        x = self.data.denormalize_data(x)
        return x

    @torch.no_grad()
    def compute_finetuned_evolution(self, x0, alpha0, steps=100):
        h = 1 / steps
        x_traj = [x0]
        alpha_traj = [alpha0]
        t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        for i in range(steps):
            t = t_steps[i]
            xt = x_traj[i].detach()
            alpha_t = alpha_traj[i].detach()

            vt_x_base = self.vt_x_base(xt, t)
            vt_alpha_base = self.vt_alpha_base(xt, alpha_t, vt_x_base, t)

            vt_x, vt_alpha = self.vt_finetune(xt, alpha_t, vt_alpha_base, t)
            xt = xt + h * vt_x
            alpha_t = alpha_t + h * vt_alpha

            x_traj.append(xt)
            alpha_traj.append(alpha_t)

        x_traj = torch.stack(x_traj, dim=1)
        alpha_traj = torch.stack(alpha_traj, dim=1)
        return x_traj, alpha_traj


    def compute_evolution_guide(self, x0, alpha0, alpha_target, target_mask=None, steps=100, guidance_x=1e-4, guidance_alpha=1e-3):
        x_traj = [x0]
        alpha_traj = [alpha0]
        t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        for i in range(steps):
            t = t_steps[i]
            h = t_steps[i+1] - t_steps[i]
            xt = x_traj[i].detach()
            alpha_t = alpha_traj[i].detach()

            xt.requires_grad_(True)
            alpha_t.requires_grad_(True)

            vt_x_base = self.vt_x_base(xt, t).detach()
            vt_alpha_base = self.vt_alpha_base(xt, alpha_t, vt_x_base, t)

            vt_x, vt_alpha = self.vt_finetune(xt, alpha_t, vt_alpha_base, t)


            alpha_1 = self.denormalize_alpha(alpha_t + (1-t) * vt_alpha)

            diff_sq = (alpha_target - alpha_1) ** 2

            spatial_dims = tuple(range(1, vt_x.ndim))
            if target_mask is not None:
                target_loss = 1 / 2 * (diff_sq * target_mask).sum(dim=spatial_dims) / target_mask.sum(dim=spatial_dims)
            else:
                target_loss = 1 / 2 * diff_sq.mean(dim=spatial_dims)

            grad_xt, grad_alpha_t = torch.autograd.grad(outputs=target_loss,
                                                        inputs=(xt, alpha_t),
                                                        grad_outputs=torch.ones_like(target_loss))

            xt = xt + h * vt_x - h * guidance_x * grad_xt
            alpha_t = alpha_t + h * vt_alpha - h * guidance_alpha * grad_alpha_t

            x_traj.append(xt.detach())
            alpha_traj.append(alpha_t.detach())

        x_traj = torch.stack(x_traj, dim=1)
        alpha_traj = torch.stack(alpha_traj, dim=1)
        return x_traj, alpha_traj


    def compute_evolution_guide_heun_alpha(self, x0, alpha0, alpha_target, target_mask=None, steps=100, guidance_x=1e-4, guidance_alpha=1e-3):
        x_traj = [x0]
        alpha_traj = [alpha0]
        t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        for i in range(steps):
            t = t_steps[i]
            t_next = t_steps[i+1]
            h = t_next - t_steps[i]
            xt = x_traj[i].detach()
            alpha_t = alpha_traj[i].detach()

            xt.requires_grad_(True)
            alpha_t.requires_grad_(True)

            # first step:
            vt_x_base = self.vt_x_base(xt, t)
            vt_alpha_base = self.vt_alpha_base(xt, alpha_t, vt_x_base, t)
            vt_x, vt_alpha = self.vt_finetune(xt, alpha_t, vt_alpha_base, t)

            xt_p1 = xt + h * vt_x
            alpha_t_p1 = alpha_t + h * vt_alpha

            if i < steps-1:
                # second step:
                vt_x_base_next = self.vt_x_base(xt_p1, t_next)
                vt_alpha_base_next = self.vt_alpha_base(xt_p1, alpha_t_p1, vt_x_base_next, t_next)
                vt_x_next, vt_alpha_next = self.vt_finetune(xt_p1, alpha_t_p1, vt_alpha_base_next, t_next)

                alpha_1_hat = alpha_t_p1 + (1 - t_next) * vt_alpha_next

                alpha_1 = self.denormalize_alpha(alpha_1_hat)

                diff_sq = (alpha_target - alpha_1) ** 2

                spatial_dims = tuple(range(1, alpha_1.ndim))
                if target_mask is not None:
                    target_loss = (diff_sq * target_mask).sum(dim=spatial_dims) / target_mask.sum(dim=spatial_dims)
                else:
                    target_loss =diff_sq.mean(dim=spatial_dims)

                grad_xt, grad_alpha_t = torch.autograd.grad(outputs=target_loss,
                                                            inputs=(xt, alpha_t),
                                                            grad_outputs=torch.ones_like(target_loss))

                xt = xt + 0.5 * h * (vt_x + vt_x_next)
                alpha_t = alpha_t + 0.5 * h * (vt_alpha + vt_alpha_next)

                scale = (1 - i / steps) ** 0.5
                xt = xt - scale * guidance_x * grad_xt
                alpha_t = alpha_t - scale * guidance_alpha * grad_alpha_t

            else:
                xt = xt_p1
                alpha_t = alpha_t_p1

            x_traj.append(xt.detach())
            alpha_traj.append(alpha_t.detach())

        x_traj = torch.stack(x_traj, dim=1)
        alpha_traj = torch.stack(alpha_traj, dim=1)
        return x_traj, alpha_traj


    def compute_evolution_guide_heun_x(self, x0, alpha0, x_target, target_mask=None, steps=100, guidance_x=1e-4, guidance_alpha=1e-3):
        x_traj = [x0]
        alpha_traj = [alpha0]
        t_steps = torch.linspace(0, 1, steps + 1).to(self.device)

        for i in range(steps):
            t = t_steps[i]
            t_next = t_steps[i+1]
            h = t_next - t_steps[i]
            xt = x_traj[i].detach()
            alpha_t = alpha_traj[i].detach()

            xt.requires_grad_(True)
            alpha_t.requires_grad_(True)

            # first step:
            vt_x_base = self.vt_x_base(xt, t)
            vt_alpha_base = self.vt_alpha_base(xt, alpha_t, vt_x_base, t)
            vt_x, vt_alpha = self.vt_finetune(xt, alpha_t, vt_alpha_base, t)

            xt_p1 = xt + h * vt_x
            alpha_t_p1 = alpha_t + h * vt_alpha

            if i < steps-1:
                # second step:
                vt_x_base_next = self.vt_x_base(xt_p1, t_next)
                vt_alpha_base_next = self.vt_alpha_base(xt_p1, alpha_t_p1, vt_x_base_next, t_next)
                vt_x_next, vt_alpha_next = self.vt_finetune(xt_p1, alpha_t_p1, vt_alpha_base_next, t_next)

                x_1_hat = xt_p1 + (1 - t_next) * vt_x_next

                diff_sq = (x_target - x_1_hat) ** 2

                spatial_dims = tuple(range(1, x_1_hat.ndim))
                if target_mask is not None:
                    target_loss = (diff_sq * target_mask).sum(dim=spatial_dims) / target_mask.sum(dim=spatial_dims)
                else:
                    target_loss = diff_sq.mean(dim=spatial_dims)

                grad_xt, grad_alpha_t = torch.autograd.grad(outputs=target_loss,
                                                            inputs=(xt, alpha_t),
                                                            grad_outputs=torch.ones_like(target_loss))

                xt = xt + 0.5 * h * (vt_x + vt_x_next)
                alpha_t = alpha_t + 0.5 * h * (vt_alpha + vt_alpha_next)

                scale = (1 - i / steps) ** 0.5
                xt = xt - scale * guidance_x * grad_xt
                alpha_t = alpha_t - scale * guidance_alpha * grad_alpha_t
            else:
                xt = xt_p1
                alpha_t = alpha_t_p1

            x_traj.append(xt.detach())
            alpha_traj.append(alpha_t.detach())

        x_traj = torch.stack(x_traj, dim=1)
        alpha_traj = torch.stack(alpha_traj, dim=1)
        return x_traj, alpha_traj