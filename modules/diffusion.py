from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["NoisePredictor", "Diffusion"]


class NoisePredictor(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1))


class Diffusion:
    def __init__(
        self,
        data_dim: int,
        steps: int = 100,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.data_dim = data_dim
        self.steps = steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.model = NoisePredictor(data_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def add_noise(self, clean: torch.Tensor, time: torch.Tensor):
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[time].unsqueeze(1)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def update(self, clean: torch.Tensor) -> float:
        clean = clean.to(self.device)
        time = torch.randint(0, self.steps, (clean.size(0),), device=self.device)
        noisy, noise = self.add_noise(clean, time)
        t = time.float().unsqueeze(1) / self.steps
        predicted_noise = self.model(noisy, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def sample_trajectory(self, batch_size: int) -> torch.Tensor:
        x = torch.randn(batch_size, self.data_dim, device=self.device)
        trajectory = [x.clone()]

        for step in reversed(range(self.steps)):
            time = torch.full((batch_size,), step, device=self.device)
            t = time.float().unsqueeze(1) / self.steps
            predicted_noise = self.model(x, t)

            beta = self.betas[time].unsqueeze(1)
            alpha = self.alphas[time].unsqueeze(1)
            alpha_bar = self.alpha_bars[time].unsqueeze(1)
            x = (x - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()

            if step > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
            trajectory.append(x.clone())

        return torch.stack(trajectory, dim=0)

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        return self.sample_trajectory(batch_size)[-1]

