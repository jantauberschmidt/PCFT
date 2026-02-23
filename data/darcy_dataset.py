import torch
from torch.utils.data import Dataset


class DarcyDataset(Dataset):
    def __init__(self, path, n_data=None, add_noise=True, return_a=False, noise_scale=1e-4, normalize=True):
        data = torch.load(path, map_location='cpu')
        self.a = data['a']
        self.u = data['u']

        self.mean_u = self.u.mean()
        self.std_u = self.u.std()

        self.param_range = (self.a.min(), self.a.max())

        if normalize:
            self.u = (self.u - self.mean_u) / self.std_u

        self.add_noise = add_noise
        self.return_a = return_a

        if normalize:
            self.noise_scale = noise_scale / self.std_u
        else:
            self.noise_scale = noise_scale


        if n_data is None:
            self.n_data = self.a.shape[0]
        else:
            self.n_data = n_data

    def return_param(self, toggle):
        self.return_a = toggle

    def denormalize_data(self, u):
        return u * self.std_u + self.mean_u

    def denormalize_alpha(self, alpha_raw):
        gate = torch.sigmoid(alpha_raw)
        alpha_pred = gate * self.param_range[1] + (1 - gate) * self.param_range[0]
        return alpha_pred


    def __len__(self):
        return min(self.n_data, self.a.shape[0])

    def __getitem__(self, idx):
        if self.add_noise:
            u = self.u[idx].unsqueeze(0)
            if self.return_a:
                return self.a[idx], u + self.noise_scale * torch.randn_like(u)
            else:
                return u + self.noise_scale * torch.randn_like(u)
        else:
            u = self.u[idx].unsqueeze(0)
            if self.return_a:
                return self.a[idx], u
            else:
                return u

