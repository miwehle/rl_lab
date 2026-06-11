"""Reusable DQN training loop for Gymnasium environments.

The Big 5 in DQN are:
- Q-function represented as neural network
- TD target
- TD error
- Target network (innovation)
- Replay memory (innovation)

They can easily be found in this module via naming and comments.


References:
    https://web.stanford.edu/class/cs234/slides/lecture4pre.pdf, p. 62-67
    https://introml.mit.edu/notes/reinforcement_learning.html#q-learning-with-function-approximation, ch. 12.2
"""

from collections.abc import Callable
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import count
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from dqn.model import DQN


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
ModelFactory = Callable[[int, int], nn.Module]

# DQN Big 5 member
class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


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

    def __post_init__(self) -> None:
        positive_counts = {
            "batch_size": self.batch_size,
            "eps_decay": self.eps_decay,
            "num_episodes": self.num_episodes,
        }
        if any(value < 1 for value in positive_counts.values()):
            raise ValueError(f"must be >= 1: {', '.join(positive_counts)}")

        probabilities = {
            "gamma": self.gamma,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "tau": self.tau,
        }
        if any(value < 0 or value > 1 for value in probabilities.values()):
            raise ValueError(f"must be between 0 and 1: {', '.join(probabilities)}")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")


@dataclass
class TrainingResult:
    q_net: nn.Module
    episode_returns: list[float]
    episode_lengths: list[int]


class Trainer:
    def __init__(
        self,
        env,
        seed: int | None = None,
        device=None,
        replay_memory_capacity: int = 10_000,
        model_factory: ModelFactory = DQN,
    ) -> None:
        self.env = env
        self.steps_done = 0
        self.device = resolve_device(device)

        if seed is not None:
            set_seeds(env, seed)

        n_actions = env.action_space.n
        state, _ = env.reset(seed=seed)
        n_observations = len(state)

        # DQN Big 5 members: Q-network and target network
        self.q_net = model_factory(n_observations, n_actions).to(self.device)
        self.target_net = model_factory(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=TrainingConfig().learning_rate,
            weight_decay=0.01,
            amsgrad=True,
        )
        self.memory = ReplayMemory(replay_memory_capacity)

    def train(self, config: TrainingConfig, plotter=None) -> TrainingResult: # NOSONAR
        """Train for the configured episodes, continuing existing trainer state."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = config.learning_rate

        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(config.num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_return = 0.0

            for t in count():
                action = self._select_action(state, config)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_return += float(reward)

                reward_tensor, observation_tensor = self._as_tensors(reward, observation)
                next_state = None if terminated else observation_tensor

                self.memory.push(state, action, next_state, reward_tensor)
                state = next_state

                if self._should_optimize(config):
                    self._optimize_model(config)
                self._soft_target_update(config.tau)

                done = terminated or truncated
                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(t + 1)

                    self._after_episode(episode_returns, episode_lengths, config, plotter)

                    break

        return TrainingResult(self.q_net, episode_returns, episode_lengths)

    def _as_tensors(self, reward, observation) -> tuple[torch.Tensor, torch.Tensor]:
        reward_tensor = torch.tensor([reward], device=self.device)
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return reward_tensor, observation_tensor

    def _should_optimize(self, config: TrainingConfig) -> bool:
        """Hook used by train() to decide whether to run a gradient update."""
        return len(self.memory) >= config.batch_size

    def _after_episode(
        self,
        episode_returns: list[float],
        episode_lengths: list[int],
        config: TrainingConfig,
        plotter=None,
    ) -> None:
        """Hook used by train() after an episode has finished."""
        if plotter is not None:
            plotter.plot_returns(episode_returns)

    def _select_action(self, state: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
        sample = random.random()
        eps_threshold = config.eps_end + (
            config.eps_start - config.eps_end
        ) * math.exp(-1.0 * self.steps_done / config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Choose the action with the highest expected reward
                return self.q_net(state).max(1).indices.view(1, 1)

        action = self.env.action_space.sample()
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def _optimize_model(self, config: TrainingConfig) -> None:
        batch = self._sample_replay_batch(config.batch_size)

        # Select Q-values for the actions that were actually taken
        q_values = self.q_net(torch.cat(batch.state)).gather(1, torch.cat(batch.action))

        next_q_values = self._next_q_values(batch.next_state, config.batch_size)

        # DQN Big 5: TD targets
        td_targets = (next_q_values * config.gamma) + torch.cat(batch.reward)

        # Huber loss on the TD error (td_target - q_values)
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, td_targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients in-place
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        # Update q_net weights to move q_values toward Bellman targets
        self.optimizer.step()

    def _sample_replay_batch(self, batch_size: int) -> Transition:
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        return Transition(*zip(*transitions))

    def _next_q_values(
        self,
        next_states: tuple[torch.Tensor | None, ...],
        batch_size: int,
    ) -> torch.Tensor:
        # Final states keep value 0
        next_q_values = torch.zeros(batch_size, device=self.device)

        # Mask states that are not terminal
        non_final_mask = torch.tensor(
            tuple(state is not None for state in next_states),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_states = [state for state in next_states if state is not None]

        if non_final_states:
            with torch.no_grad():
                # Estimate next-state values with the older target network
                next_q_values[non_final_mask] = self.target_net(
                    torch.cat(non_final_states)
                ).max(1).values

        return next_q_values

    def _soft_target_update(self, tau: float) -> None:
        target = self.target_net.state_dict()
        policy = self.q_net.state_dict()

        for key in policy:
            # target is an EMA of policy
            target[key] = tau * policy[key] + (1 - tau) * target[key]

        self.target_net.load_state_dict(target)


def resolve_device(device=None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(env, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
