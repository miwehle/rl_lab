"""Reusable DQN training loop for Gymnasium environments."""

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
    policy_net: DQN
    episode_returns: list[float]
    episode_lengths: list[int]
    device: torch.device


class Trainer:
    def __init__(self, env, seed: int | None = None, device=None, replay_memory_capacity: int = 10_000) -> None:
        self.env = env
        self.seed = seed
        self.steps_done = 0
        self.device = resolve_device(device)

        if seed is not None:
            set_seeds(env, seed)

        n_actions = env.action_space.n
        state, _ = env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=TrainingConfig().learning_rate,
            amsgrad=True,
        )
        self.memory = ReplayMemory(replay_memory_capacity)

    def train(self, config: TrainingConfig, plotter=None) -> TrainingResult:
        self.config = config
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = config.learning_rate

        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        for episode_index in range(config.num_episodes):
            reset_seed = self.seed if episode_index == 0 and self.steps_done == 0 else None
            state, _ = self.env.reset(seed=reset_seed)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_return = 0.0
            steps = range(config.max_steps_per_episode) if config.max_steps_per_episode is not None else count()

            for step_index in steps:
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_return += float(reward)
                reward_tensor = torch.tensor([reward], device=self.device)

                step_limit_reached = (
                    config.max_steps_per_episode is not None
                    and step_index + 1 >= config.max_steps_per_episode
                )
                done = terminated or truncated or step_limit_reached

                if terminated or step_limit_reached:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward_tensor)
                state = next_state

                self.optimize_model()
                self.soft_update()

                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(step_index + 1)

                    if plotter is not None:
                        plotter.plot_returns(episode_returns)

                    break

        return TrainingResult(self.policy_net, episode_returns, episode_lengths, self.device)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        action = self.env.action_space.sample()
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self) -> None:
        if len(self.memory) < self.config.batch_size:
            return

        transitions = self.memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(state is not None for state in batch.next_state), device=self.device, dtype=torch.bool)
        non_final_states = [state for state in batch.next_state if state is not None]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.config.batch_size, device=self.device)
        if non_final_states:
            non_final_next_states = torch.cat(non_final_states)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self) -> None:
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # θ′ ← τ θ + (1 −τ )θ′
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.tau + target_net_state_dict[key] * (1 - self.config.tau)

        self.target_net.load_state_dict(target_net_state_dict)


def resolve_device(device=None) -> torch.device:
    return device or torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def set_seeds(env, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
