import torch
import torch.nn as nn

class SaBN2d(nn.Module):
    """
    Sandwich Batch Norm for 3D feature maps, conditioned on a per-sample
    categorical label (e.g. site/scanner id).
    """
    def __init__(self, num_features, num_conditions, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_conditions = num_conditions
        # normalization only — no built-in affine, we supply our own
        self.norm = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)

        # shared "sandwich" affine — same for every site
        self.shared_gamma = nn.Parameter(torch.ones(num_features))
        self.shared_beta = nn.Parameter(torch.zeros(num_features))

        # independent per-site affine layers
        self.site_gamma = nn.Embedding(num_conditions, num_features)
        self.site_beta = nn.Embedding(num_conditions, num_features)
        nn.init.ones_(self.site_gamma.weight)
        nn.init.zeros_(self.site_beta.weight)

    def forward(self, x, site_id):
        # x: (B, C, D, H, W)   site_id: (B,) long tensor
        x_hat = self.norm(x)

        shared_g = self.shared_gamma.view(1, -1, 1, 1, 1)
        shared_b = self.shared_beta.view(1, -1, 1, 1, 1)
        x_sa = shared_g * x_hat + shared_b

        g_i = self.site_gamma(site_id).view(-1, x.shape[1], 1, 1, 1)
        b_i = self.site_beta(site_id).view(-1, x.shape[1], 1, 1, 1)

        return g_i * x_sa + b_i
    
class SaBN3d(nn.Module):
    """
    Sandwich Batch Norm for 3D feature maps, conditioned on a per-sample
    categorical label (e.g. site/scanner id).
    """
    def __init__(self, num_features, num_conditions, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_conditions = num_conditions
        # normalization only — no built-in affine, we supply our own
        self.norm = nn.BatchNorm3d(num_features, affine=False, eps=eps, momentum=momentum)

        # shared "sandwich" affine — same for every site
        self.shared_gamma = nn.Parameter(torch.ones(num_features))
        self.shared_beta = nn.Parameter(torch.zeros(num_features))

        # independent per-site affine layers
        self.site_gamma = nn.Embedding(num_conditions, num_features)
        self.site_beta = nn.Embedding(num_conditions, num_features)
        nn.init.ones_(self.site_gamma.weight)
        nn.init.zeros_(self.site_beta.weight)

    def forward(self, x, site_id):
        # x: (B, C, D, H, W)   site_id: (B,) long tensor
        x_hat = self.norm(x)

        shared_g = self.shared_gamma.view(1, -1, 1, 1, 1)
        shared_b = self.shared_beta.view(1, -1, 1, 1, 1)
        x_sa = shared_g * x_hat + shared_b

        g_i = self.site_gamma(site_id).view(-1, x.shape[1], 1, 1, 1)
        b_i = self.site_beta(site_id).view(-1, x.shape[1], 1, 1, 1)

        return g_i * x_sa + b_i