import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("time_dim must be even and at least four")
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim * 2))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, t):
        scale, shift = self.time_mlp(t).chunk(2, dim=-1)
        return x + self.net(x * (scale + 1) + shift)


class TinyDiffusion(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, depth=6, time_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_dim) for _ in range(depth)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )
        self.alni_scales = nn.Parameter(torch.ones(depth))

    def forward(self, x, t):
        h = self.input_proj(x)
        te = self.time_embed(t)
        for i, block in enumerate(self.blocks):
            noise_inject = torch.zeros_like(h)
            if self.training:
                noise_inject = (
                    self.alni_scales[i]
                    * torch.randn_like(h)
                    * (t.float() / 1000).unsqueeze(-1)
                )
            h = block(h + noise_inject, te)
        return self.output_proj(h)
