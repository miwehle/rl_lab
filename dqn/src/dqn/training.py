"""Reusable DQN training loop for Gymnasium environments."""

from collections.abc import Callable
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import count
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from dqn.config import TrainingConfig
from dqn.model import DQN


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
ModelFactory = Callable[[int, int], nn.Module]


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
class TrainingResult:
    policy_net: nn.Module
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

        self.policy_net = model_factory(n_observations, n_actions).to(self.device)
        self.target_net = model_factory(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
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
                action = self.select_action(state, config)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_return += float(reward)
                reward_tensor = torch.tensor([reward], device=self.device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                self.memory.push(state, action, next_state, reward_tensor)
                state = next_state

                self.optimize_model(config)
                self.soft_target_update(config.tau)

                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(t + 1)

                    if plotter is not None:
                        plotter.plot_returns(episode_returns)

                    break

        return TrainingResult(self.policy_net, episode_returns, episode_lengths)

    def select_action(self, state: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
        sample = random.random()
        eps_threshold = config.eps_end + (
            config.eps_start - config.eps_end
        ) * math.exp(-1.0 * self.steps_done / config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Choose the action with the highest expected reward
                return self.policy_net(state).max(1).indices.view(1, 1)

        action = self.env.action_space.sample()
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self, config: TrainingConfig) -> None:
        if len(self.memory) < config.batch_size:
            return

        transitions = self.memory.sample(config.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Mask states that are not terminal
        non_final_mask = torch.tensor(
            tuple(state is not None for state in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_states = [state for state in batch.next_state if state is not None]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Select Q-values for the actions that were actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Final states keep value 0
        next_state_values = torch.zeros(config.batch_size, device=self.device)
        if non_final_states:
            non_final_next_states = torch.cat(non_final_states)
            with torch.no_grad():
                # Estimate next-state values with the older target network
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1).values

        # Bellman target
        expected_state_action_values = (next_state_values * config.gamma) + reward_batch

        # Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients in-place
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_target_update(self, tau: float) -> None:
        target = self.target_net.state_dict()
        policy = self.policy_net.state_dict()

        # target is an EMA of policy
        for key in policy:
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
