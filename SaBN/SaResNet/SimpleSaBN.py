import torch
import torch.nn as nn


class SaBN2d(nn.Module):
    """
    Sandwich Batch Norm for 2D feature maps, conditioned on a per-sample
    categorical label (e.g. site/scanner id).
    """
    def __init__(self, num_features, num_conditions=1, eps=1e-5, momentum=0.1, **kwargs):
        super().__init__()
        self.num_conditions = num_conditions
        self.norm = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)

        # shared "sandwich" affine — same for every site
        self.shared_gamma = nn.Parameter(torch.ones(num_features))
        self.shared_beta = nn.Parameter(torch.zeros(num_features))

        # independent per-site affine layers
        self.cond_gamma = nn.Embedding(num_conditions, num_features)
        self.cond_beta = nn.Embedding(num_conditions, num_features)
        # FIX: was referencing self.site_gamma / self.site_beta, which are
        # never defined — the attributes created above are cond_gamma/cond_beta
        nn.init.ones_(self.cond_gamma.weight)
        nn.init.zeros_(self.cond_beta.weight)

    def forward(self, x, cond):
        # x: (B, C, H, W)   cond: (B,) long tensor
        x_hat = self.norm(x)

        # FIX: was .view(1, -1, 1, 1, 1) (5D, copy-pasted from the 3D
        # version) against a 4D (B, C, H, W) tensor -> shape-mismatch error.
        shared_g = self.shared_gamma.view(1, -1, 1, 1)
        shared_b = self.shared_beta.view(1, -1, 1, 1)
        x_sa = shared_g * x_hat + shared_b

        if cond == "U":
            return x_sa
        
        g_i = self.cond_gamma(cond).view(-1, x.shape[1], 1, 1)
        b_i = self.cond_beta(cond).view(-1, x.shape[1], 1, 1)

        return g_i * x_sa + b_i


class SaBN3d(nn.Module):
    """
    Sandwich Batch Norm for 3D feature maps, conditioned on a per-sample
    categorical label (e.g. site/scanner id).
    """
    def __init__(self, num_features, num_conditions=1, eps=1e-5, momentum=0.1, **kwargs):
        super().__init__()
        self.num_conditions = num_conditions
        self.norm = nn.BatchNorm3d(num_features, affine=False, eps=eps, momentum=momentum)

        self.shared_gamma = nn.Parameter(torch.ones(num_features))
        self.shared_beta = nn.Parameter(torch.zeros(num_features))

        self.cond_gamma = nn.Embedding(num_conditions, num_features)
        self.cond_beta = nn.Embedding(num_conditions, num_features)
        # FIX: same self.site_gamma / self.site_beta typo as SaBN2d
        nn.init.ones_(self.cond_gamma.weight)
        nn.init.zeros_(self.cond_beta.weight)

    def forward(self, x, cond):
        # x: (B, C, D, H, W)   cond: (B,) long tensor
        x_hat = self.norm(x)

        shared_g = self.shared_gamma.view(1, -1, 1, 1, 1)
        shared_b = self.shared_beta.view(1, -1, 1, 1, 1)
        x_sa = shared_g * x_hat + shared_b

        if cond == "U":
            return x_sa
        
        g_i = self.cond_gamma(cond).view(-1, x.shape[1], 1, 1, 1)
        b_i = self.cond_beta(cond).view(-1, x.shape[1], 1, 1, 1)

        return g_i * x_sa + b_i