import torch
from torch.utils.data import Dataset


class HelmholtzDataset(Dataset):
    def __init__(self, path, n_data=None, return_c=False, normalize=True):
        data = torch.load(path, map_location='cpu')
        self.c = data['c']
        self.u = data['u']

        self.return_c = return_c

        self.mean_u = self.u.mean()
        self.std_u = self.u.std()

        self.param_range = (self.c.min(), self.c.max())

        if normalize:
            self.u = (self.u - self.mean_u) / self.std_u


        if n_data is None:
            self.n_data = self.c.shape[0]
        else:
            self.n_data = n_data

    def return_param(self, toggle):
        self.return_c = toggle

    def denormalize_data(self, u):
        return u * self.std_u + self.mean_u

    def denormalize_alpha(self, alpha_raw):
        gate = torch.sigmoid(alpha_raw)
        alpha_pred = gate * self.param_range[1] + (1 - gate) * self.param_range[0]
        return alpha_pred


    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        u = self.u[idx]
        if self.return_c:
            return self.c[idx], u
        else:
            return u