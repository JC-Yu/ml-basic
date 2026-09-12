from __future__ import annotations

import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

__all__ = ["Actor", "Critic", "ReplayBuffer", "SACAgent"]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
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
        return self.mean_head(h), self.std_head(h)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int):
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

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32),
            torch.as_tensor(np.stack(actions), dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 1_000
    ):
        self.max_action = float(max_action)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        self.replay_buffer = ReplayBuffer(buffer_size)

    def _sample_action(self, state):
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action * self.max_action, log_prob.sum(dim=-1)

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.max_action
            else:
                action = self._sample_action(state)[0]
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_states)
            target_q = torch.min(
                self.target_critic1(next_states, next_actions),
                self.target_critic2(next_states, next_actions),
            ).squeeze(1) - self.alpha * next_log_probs
            targets = rewards + self.gamma * (1.0 - dones) * target_q

        q1 = self.critic1(states, actions).squeeze(1)
        q2 = self.critic2(states, actions).squeeze(1)
        critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self._sample_action(states)
        q = torch.min(
            self.critic1(states, new_actions),
            self.critic2(states, new_actions),
        ).squeeze(1)
        actor_loss = (self.alpha * log_probs - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

        return float(critic_loss.item())
