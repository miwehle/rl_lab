"""Reusable DQN training loop for Gymnasium environments."""

from __future__ import annotations

from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import count
import math
import random
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.config import TrainingConfig
from dqn.model import DQN


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
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


def resolve_device(config: TrainingConfig) -> torch.device:
    if config.device is not None:
        return config.device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seeds(env: gym.Env, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def validate_env(env: gym.Env) -> tuple[int, int]:
    action_space = env.action_space
    observation_space = env.observation_space

    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError("DQN training requires a discrete action space.")

    if not isinstance(observation_space, gym.spaces.Box):
        raise TypeError("DQN training requires a Box observation space.")

    if len(observation_space.shape) != 1:
        raise ValueError("DQN training requires a flat observation space with shape=(n,).")

    return observation_space.shape[0], int(action_space.n)


def select_action(
    state: torch.Tensor,
    policy_net: DQN,
    action_space: gym.spaces.Discrete,
    device: torch.device,
    steps_done: int,
    config: TrainingConfig,
) -> tuple[torch.Tensor, int]:
    sample = random.random()
    eps_threshold = config.eps_end + (config.eps_start - config.eps_end) * math.exp(
        -1.0 * steps_done / config.eps_decay
    )
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)
            return action, steps_done

    action = torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)
    return action, steps_done


def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    memory: ReplayMemory,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    if len(memory) < config.batch_size:
        return

    transitions = memory.sample(config.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(state is not None for state in batch.next_state),
        device=device,
        dtype=torch.bool,
    )
    non_final_states = [state for state in batch.next_state if state is not None]

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(config.batch_size, device=device)
    if non_final_states:
        non_final_next_states = torch.cat(non_final_states)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * config.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def soft_update(policy_net: DQN, target_net: DQN, tau: float) -> None:
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = (
            policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        )

    target_net.load_state_dict(target_net_state_dict)


def train(env: gym.Env, config: TrainingConfig, plotter: Any = None) -> TrainingResult:
    n_observations, n_actions = validate_env(env)
    device = resolve_device(config)

    if config.seed is not None:
        set_seeds(env, config.seed)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=config.learning_rate, amsgrad=True)
    memory = ReplayMemory(config.replay_memory_capacity)

    steps_done = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for episode_index in range(config.num_episodes):
        reset_seed = config.seed if episode_index == 0 else None
        state, _ = env.reset(seed=reset_seed)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_return = 0.0

        step_iterable = count()
        if config.max_steps_per_episode is not None:
            step_iterable = range(config.max_steps_per_episode)

        for step_index in step_iterable:
            action, steps_done = select_action(
                state=state,
                policy_net=policy_net,
                action_space=env.action_space,
                device=device,
                steps_done=steps_done,
                config=config,
            )

            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_return += float(reward)
            reward_tensor = torch.tensor([reward], device=device)

            step_limit_reached = (
                config.max_steps_per_episode is not None
                and step_index + 1 >= config.max_steps_per_episode
            )
            done = terminated or truncated or step_limit_reached

            if terminated or step_limit_reached:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            optimize_model(policy_net, target_net, optimizer, memory, config, device)
            soft_update(policy_net, target_net, config.tau)

            if done:
                episode_returns.append(episode_return)
                episode_lengths.append(step_index + 1)

                if plotter is not None:
                    plotter.plot_returns(episode_returns)

                break

    return TrainingResult(
        policy_net=policy_net,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        device=device,
    )

