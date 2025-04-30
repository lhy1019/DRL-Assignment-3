"""
models.py
==========

QNetwork – DeepMind (2015) convolutional architecture
Input :  (B, 4, 84, 84)  uint8 / float32
Output:  (B, 12)         Q-values for COMPLEX_MOVEMENT

Usage
-----
from models import QNetwork
net = QNetwork(in_channels=4, n_actions=12)
q   = net(torch.zeros(1,4,84,84))   # torch.Size([1,12])
"""
import torch, math
import torch.nn as nn
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f

        self.weight_mu    = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer("weight_eps", torch.empty(out_f, in_f))

        self.bias_mu    = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer("bias_eps", torch.empty(out_f))

        self.reset_parameters(sigma_init)
        self.reset_noise()
        
    def reset_parameters(self, sigma_init):
        mu_range = 1.0 / math.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_f))
        self.bias_sigma.data.fill_(sigma_init)
        
    @staticmethod
    def _scale_noise(size, device):
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())          # factorised trick

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_f,  self.weight_mu.device)
        eps_out = self._scale_noise(self.out_f, self.weight_mu.device)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:                      # deterministic in eval mode
            w, b = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, w, b)
    
    
class DuelingQNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()

        # ── shared convolutional backbone ──────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32,          64, 4, 2), nn.ReLU(),
            nn.Conv2d(64,          64, 3, 1), nn.ReLU()
        )

        # flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84, device=self.conv[0].weight.device)

            flat  = self.conv(dummy).view(1, -1)
            feat_dim = flat.size(1)

        # ── value & advantage streams ─────────────────────────────────
        self.value_head = nn.Sequential(
            NoisyLinear(feat_dim, 512), nn.ReLU(),
            NoisyLinear(512, 1)                  # V(s)
        )

        self.adv_head = nn.Sequential(
            NoisyLinear(feat_dim, 512), nn.ReLU(),
            NoisyLinear(512, n_actions)           # A(s,a)
        )

    # -----------------------------------------------------------------
    def forward(self, x):
        # allow uint8 inputs straight from replay buffer
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        feats = self.conv(x).flatten(1)               # (B, feat_dim)

        value = self.value_head(feats)                # (B, 1)
        adv   = self.adv_head(feats)                  # (B, n_actions)

        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q
    
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

