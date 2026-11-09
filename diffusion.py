import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        if timesteps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Expected a positive step count and betas in (0, 1)")
        self.T = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register(betas, "betas")
        self.register(alpha_bar, "alpha_bar")
        self.register(torch.sqrt(alpha_bar), "sqrt_alpha_bar")
        self.register(torch.sqrt(1.0 - alpha_bar), "sqrt_one_minus_alpha_bar")
        self.register(1.0 / torch.sqrt(alphas), "recip_sqrt_alpha")
        self.register(betas / torch.sqrt(1.0 - alpha_bar), "betas_over_sigma")

    def register(self, tensor, name):
        setattr(self, name, tensor)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t].view(-1, *([1] * (x0.ndim - 1)))
        s = self.sqrt_one_minus_alpha_bar[t].view(-1, *([1] * (x0.ndim - 1)))
        return a * x0 + s * noise, noise

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            coef = self.betas_over_sigma[i]
            x = self.recip_sqrt_alpha[i] * (x - coef * pred_noise)
            if i > 0:
                x += torch.sqrt(self.betas[i]) * torch.randn_like(x)
        return x
