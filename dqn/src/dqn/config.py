"""Training configuration for DQN runs."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 128
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 2500
    tau: float = 0.005
    learning_rate: float = 3e-4
    num_episodes: int = 50
