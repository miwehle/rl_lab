"""Neural network model for flat-observation DQN agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_size: int = 128,
        hidden_sizes: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or (hidden_size, hidden_size)
        if len(hidden_sizes) != 2:
            raise ValueError("hidden_sizes must contain exactly two values")

        self.layer1 = nn.Linear(n_observations, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
