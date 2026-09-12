from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["Actor", "Critic", "ReplayBuffer", "PPOAgent"]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 128):
        super().__init__()
        self.max_action = float(max_action)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),
        )

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        mean = self.mean_head(h) * self.max_action
        std = self.std_head(h)
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def as_tensors(self, device):
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32),
            torch.as_tensor(np.stack(actions), dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def clear(self):
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        lr: float = 3e-4,
        update_epochs: int = 8,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
    ):
        self.max_action = float(max_action)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_new = Actor(state_dim, action_dim, self.max_action, hidden_dim).to(self.device)
        self.actor_old = Actor(state_dim, action_dim, self.max_action, hidden_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor_new.state_dict())
        self.actor_old.eval()

        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_new.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, deterministic: bool = False):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor_old(state)
            if deterministic:
                action = mean
            else:
                action = torch.distributions.Normal(mean, std).sample()
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _gae(self, rewards, dones, values, next_values):
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            delta = rewards[idx] + self.gamma * (1.0 - dones[idx]) * next_values[idx] - values[idx]
            gae = delta + self.gamma * self.lam * (1.0 - dones[idx]) * gae
            advantages[idx] = gae
        returns = advantages + values
        return advantages, returns

    def update(self):
        if len(self.replay_buffer) == 0:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.as_tensors(self.device)
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)    # [N,]
            next_values = self.critic(next_states).squeeze(-1)  # [N,]
            advantages, returns = self._gae(rewards, dones, values, next_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            old_mean, old_std = self.actor_old(states)
            old_dist = torch.distributions.Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1)

        actor_losses = []
        critic_losses = []
        total_size = states.shape[0]

        for _ in range(self.update_epochs):
            indices = torch.randperm(total_size, device=self.device)
            for start in range(0, total_size, self.batch_size):
                mb = indices[start : start + self.batch_size]

                mean, std = self.actor_new(states[mb])
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions[mb]).sum(dim=-1)  # [B,]
                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                unclipped = ratio * advantages[mb]
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[mb]
                entropy = dist.entropy().sum(dim=-1)    # [B,]
                actor_loss = -(torch.min(unclipped, clipped)).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                value_pred = self.critic(states[mb]).squeeze(-1)
                critic_loss = nn.functional.mse_loss(value_pred, returns[mb])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))

        self.actor_old.load_state_dict(self.actor_new.state_dict())
        self.replay_buffer.clear()
        return float(np.mean(actor_losses)), float(np.mean(critic_losses))
