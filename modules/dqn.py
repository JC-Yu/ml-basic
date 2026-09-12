from __future__ import annotations

import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ["QNetwork", "ReplayBuffer", "DQNAgent"]


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
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
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        learning_starts: int = 100,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2_000,
        use_double_dqn: bool = True,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0
        self.epsilon = self.epsilon_start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_double_dqn = use_double_dqn
        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_double_dqn:
                # Use Double DQN: select actions using the online network, evaluate using the target network
                next_actions = self.q_network(next_states).argmax(dim=1)
            else:
                # Use standard DQN: select actions using the target network
                next_actions = self.target_network(next_states).argmax(dim=1)

            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1
        ratio = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())
