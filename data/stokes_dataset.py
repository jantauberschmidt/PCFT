from torch.utils.data import Dataset
import torch


class StokesDataset(Dataset):
    def __init__(self, path, n_data=None, return_nu=False, normalize=True):
        """
        Dataset for Stokes lid-driven cavity:
        - data['u']: (N, 3, H, W)  channels = (u_x, u_y, p)
        - data['nu']: (N, H, W)   viscosity field

        Parameters:
            return_nu: return viscosity field
        """
        data = torch.load(path, map_location='cpu')

        self.nu = data['nu']              # (N, H, W)
        self.u  = data['u']               # (N, 3, H, W)

        self.return_nu = return_nu          # legacy naming, now returns nu

        # ---- Per-channel normalization over (N, H, W) ----
        # mean/std shapes: (1, 3, 1, 1)
        self.mean_u = self.u.mean(dim=(0, 2, 3), keepdim=True)
        self.std_u  = self.u.std( dim=(0, 2, 3), keepdim=True)

        # Parameter range for inverse mapping (nu prediction)
        self.param_range = (self.nu.min(), self.nu.max())

        if normalize:
            self.u = (self.u - self.mean_u) / (self.std_u + 1e-8)

        # Subset handling
        if n_data is None:
            self.n_data = self.nu.shape[0]
        else:
            self.n_data = n_data

    def return_param(self, toggle):
        self.return_nu = toggle

    def denormalize_data(self, u):
        """Undo per-channel normalization."""
        return u * (self.std_u.to(u.device) + 1e-8) + self.mean_u.to(u.device)

    def denormalize_alpha(self, alpha_raw):
        """
        Map raw network outputs to viscosity values via sigmoid gate.
        Output is in [nu_min, nu_max].
        """
        gate = torch.sigmoid(alpha_raw)
        nu_pred = gate * self.param_range[1] + (1 - gate) * self.param_range[0]
        return nu_pred

    # -----------------------
    #  PyTorch dataset API
    # -----------------------

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        u = self.u[idx]
        if self.return_nu:        # legacy API, returns viscosity now
            return self.nu[idx], u
        else:
            return u
