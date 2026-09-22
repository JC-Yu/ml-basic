from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["PolicyNetwork", "REINFORCEAgent"]


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class REINFORCEAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
    ):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state, deterministic: bool = False) -> int:
        state = torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        logits = self.policy(state)

        if deterministic:
            return int(logits.argmax(dim=1).item())

        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action).squeeze(0))
        return int(action.item())

    def update(self) -> float:
        returns = []
        total_return = 0.0
        for reward in reversed(self.rewards):
            total_return = reward + self.gamma * total_return
            returns.insert(0, total_return)

        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        return float(loss.item())
