from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["Actor", "Critic", "ReplayBuffer", "TRPOAgent"]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 128):
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state: torch.Tensor):
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    def __init__(self):
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque()

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

    def clear(self):
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class TRPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lam: float = 0.95,
        max_kl: float = 0.01,
        damping: float = 0.1,
        critic_lr: float = 1e-3,
        critic_steps: int = 10,
        cg_steps: int = 10,
        line_search_steps: int = 10,
    ):
        self.max_action = float(max_action)
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.damping = damping
        self.critic_steps = critic_steps
        self.cg_steps = cg_steps
        self.line_search_steps = line_search_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.max_action, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state)
            if deterministic:
                action = torch.tanh(mean) * self.max_action
            else:
                raw_action = torch.distributions.Normal(mean, std).sample()
                action = torch.tanh(raw_action) * self.max_action
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

    def _flat_params(self, model: nn.Module) -> torch.Tensor:
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    def _set_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> None:
        start = 0
        for param in model.parameters():
            size = param.numel()
            param.data.copy_(flat_params[start : start + size].view_as(param))
            start += size

    def _flat_grad(self, loss, model: nn.Module, retain_graph=False, create_graph=False):
        params = list(model.parameters())
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        return torch.cat(
            [
                torch.zeros_like(param).view(-1) if grad is None else grad.contiguous().view(-1)
                for param, grad in zip(params, grads)
            ]
        )

    def _log_prob(self, states, actions):
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        squashed = (actions / self.max_action).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(squashed)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(self.max_action * (1.0 - squashed.pow(2)) + 1e-6)
        return log_prob.sum(dim=-1)

    def _surrogate(self, states, actions, old_log_probs, advantages):
        log_probs = self._log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def _mean_kl(self, states, old_mean, old_std):
        mean, std = self.actor(states)
        old_dist = torch.distributions.Normal(old_mean, old_std)
        new_dist = torch.distributions.Normal(mean, std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1).mean()

    def _hessian_vector_product(self, states, old_mean, old_std, vector):
        kl = self._mean_kl(states, old_mean, old_std)
        grad_kl = self._flat_grad(kl, self.actor, retain_graph=True, create_graph=True)
        grad_vector = (grad_kl * vector).sum()
        hvp = self._flat_grad(grad_vector, self.actor)
        return hvp + self.damping * vector

    def _conjugate_gradient(self, hvp_fn, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_r = torch.dot(r, r)

        for _ in range(self.cg_steps):
            hvp = hvp_fn(p)
            alpha = r_dot_r / (torch.dot(p, hvp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * hvp
            new_r_dot_r = torch.dot(r, r)
            if new_r_dot_r.sqrt() < 1e-10:
                break
            p = r + (new_r_dot_r / (r_dot_r + 1e-8)) * p
            r_dot_r = new_r_dot_r
        return x

    def update(self):
        if len(self.replay_buffer) == 0:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.as_tensors(self.device)
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantages, returns = self._gae(rewards, dones, values, next_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            old_mean, old_std = self.actor(states)
            old_log_probs = self._log_prob(states, actions)

        for _ in range(self.critic_steps):
            value_loss = nn.functional.mse_loss(self.critic(states).squeeze(-1), returns)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        objective = self._surrogate(states, actions, old_log_probs, advantages)
        policy_grad = self._flat_grad(objective, self.actor, retain_graph=True)
        hvp_fn = lambda v: self._hessian_vector_product(states, old_mean, old_std, v)
        step_direction = self._conjugate_gradient(hvp_fn, policy_grad)
        step_hvp = hvp_fn(step_direction)
        step_size = torch.dot(step_direction, step_hvp).clamp_min(1e-8)
        step_scale = torch.sqrt(2.0 * self.max_kl / step_size)
        full_step = step_direction * step_scale

        old_params = self._flat_params(self.actor)
        old_objective = objective.detach()
        accepted_objective = old_objective
        accepted_kl = torch.tensor(0.0, device=self.device)

        for step in range(self.line_search_steps):
            fraction = 0.5**step
            self._set_flat_params(self.actor, old_params + fraction * full_step)
            with torch.no_grad():
                new_objective = self._surrogate(states, actions, old_log_probs, advantages)
                kl = self._mean_kl(states, old_mean, old_std)
            if kl <= self.max_kl and new_objective >= old_objective:
                accepted_objective = new_objective
                accepted_kl = kl
                break
        else:
            self._set_flat_params(self.actor, old_params)

        self.replay_buffer.clear()
        return float(-accepted_objective.item()), float(value_loss.item()), float(accepted_kl.item())
