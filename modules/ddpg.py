from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

__all__ = ["Actor", "Critic", "ReplayBuffer", "DDPGAgent"]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 128):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))


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


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 1_000
    ):
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.max_action, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, noise_std: float = 0.1) -> np.ndarray:
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=self.action_dim)
        action = np.clip(action, -self.max_action, self.max_action)
        return action.astype(np.float32)

    def update(self):
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + self.gamma * (1.0 - dones) * self.critic_target(
                next_states, next_actions
            ).squeeze(1)

        current_q = self.critic(states, actions).squeeze(1)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

        return float(critic_loss.item())
