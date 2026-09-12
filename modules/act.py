from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["ActionChunkPolicy", "TemporalEnsembler", "ACTAgent"]


class ActionChunkPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_len: int,
        hidden_dim: int = 128,
        action_scale: float = 0.08,
    ):
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.chunk_len * self.action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        chunk = self.net(obs).view(-1, self.chunk_len, self.action_dim)
        return self.action_scale * torch.tanh(chunk)


class TemporalEnsembler:
    def __init__(self, chunk_len: int, decay: float = 0.8):
        self.chunk_len = int(chunk_len)
        self.decay = float(decay)
        self.reset()

    def reset(self):
        self.t = 0
        self.history = []

    def combine(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        self.history.append((self.t, chunk))

        actions = []
        weights = []
        for start, past_chunk in self.history:
            offset = self.t - start
            if 0 <= offset < self.chunk_len:
                actions.append(past_chunk[offset])
                weights.append(self.decay**offset)

        self.t += 1
        self.history = [(start, past_chunk) for start, past_chunk in self.history if self.t - start < self.chunk_len]

        if not actions:
            return chunk[0]
        return np.average(np.stack(actions), axis=0, weights=np.asarray(weights))


class ACTAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_len: int = 8,
        hidden_dim: int = 128,
        action_scale: float = 0.08,
        ensemble_decay: float = 0.8,
        lr: float = 1e-3
    ):
        self.chunk_len = int(chunk_len)
        self.action_scale = float(action_scale)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = ActionChunkPolicy(
            obs_dim,
            action_dim,
            chunk_len=self.chunk_len,
            hidden_dim=hidden_dim,
            action_scale=self.action_scale,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.ensembler = TemporalEnsembler(self.chunk_len, ensemble_decay)

    def reset(self):
        self.ensembler.reset()

    def predict_chunk(self, obs):
        obs = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            chunk = self.policy(obs).squeeze(0).cpu().numpy()
        return chunk.astype(np.float32)

    def select_action(self, obs, ensemble: bool = True):
        chunk = self.predict_chunk(obs)
        action = self.ensembler.combine(chunk) if ensemble else chunk[0]
        return np.asarray(action, dtype=np.float32), chunk

    def update(self, obs_batch, chunk_batch) -> float:
        obs_batch = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
        chunk_batch = torch.as_tensor(np.asarray(chunk_batch, dtype=np.float32), device=self.device)
        pred = self.policy(obs_batch)
        loss = nn.functional.mse_loss(pred, chunk_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
