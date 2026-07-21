"""Student models for DQN distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentDQN(nn.Module):
    """DQN MLP with independently configurable hidden layer sizes."""

    def __init__(self, n_observations: int, n_actions: int, hidden_sizes: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        if len(hidden_sizes) != 2:
            raise ValueError("hidden_sizes must contain exactly two values")
        if hidden_sizes[0] < 1 or hidden_sizes[1] < 1:
            raise ValueError("hidden_sizes must be positive")

        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.layer1 = nn.Linear(n_observations, self.hidden_sizes[0])
        self.layer2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.layer3 = nn.Linear(self.hidden_sizes[1], n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
