import torch
from torch.utils.data import Dataset


class ElasticityDataset(Dataset):
    def __init__(self, path, n_data=None, return_E=False, normalize=True):
        data = torch.load(path, map_location='cpu')
        self.E = data['E']
        self.u = data['u']

        self.u = self.u.permute(0, 3, 1, 2)
        self.E = self.E.unsqueeze(1)

        self.mean_u = self.u.mean()
        self.std_u = self.u.std()

        self.param_range = (self.E.min(), self.E.max())

        if normalize:
            self.u = (self.u - self.mean_u) / self.std_u

        self.return_E = return_E


        if n_data is None:
            self.n_data = self.E.shape[0]
        else:
            self.n_data = n_data

    def return_param(self, toggle):
        self.return_E = toggle

    def denormalize_data(self, u):
        return u * self.std_u + self.mean_u


    def denormalize_alpha(self, alpha_raw):
        gate = torch.sigmoid(alpha_raw)
        alpha_pred = gate * self.param_range[1] + (1 - gate) * self.param_range[0]
        return alpha_pred


    def __len__(self):
        return min(self.n_data, self.E.shape[0])

    def __getitem__(self, idx):
        if self.return_E:
            return self.E[idx], self.u[idx]
        else:
            return self.u[idx]
