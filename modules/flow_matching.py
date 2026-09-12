from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["FlowField", "FlowMatching"]


class FlowField(nn.Module):
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


class FlowMatching:
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        prior_std: float = 1.0,
    ):
        self.data_dim = data_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prior_std = prior_std
        self.model = FlowField(data_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update(self, source: torch.Tensor, target: torch.Tensor) -> float:
        source = source.to(self.device)
        target = target.to(self.device)
        t = torch.rand(source.size(0), 1, device=self.device)
        x = (1.0 - t) * source + t * target
        velocity = target - source
        predicted_velocity = self.model(x, t)
        loss = nn.functional.mse_loss(predicted_velocity, velocity)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def sample_trajectory(self, batch_size: int, steps: int = 50) -> torch.Tensor:
        x = torch.randn(batch_size, self.data_dim, device=self.device) * self.prior_std
        path = [x.clone()]
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((batch_size, 1), step / steps, device=self.device)
            velocity = self.model(x, t)
            x = x + velocity * dt
            path.append(x.clone())
        return torch.stack(path, dim=0)

    @torch.no_grad()
    def sample(self, batch_size: int, steps: int = 50) -> torch.Tensor:
        return self.sample_trajectory(batch_size, steps)[-1]
