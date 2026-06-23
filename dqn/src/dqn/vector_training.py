"""Vectorized DQN training for many flat-observation Gymnasium environments."""

from dataclasses import dataclass
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.model import DQN
from dqn.training import ModelFactory, TrainingConfig, TrainingResult, resolve_device


@dataclass(kw_only=True)
class VectorTrainingConfig(TrainingConfig):
    """Training config for batched environment collection."""

    learning_starts: int = 0
    optimize_every: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.optimize_every < 1:
            raise ValueError("optimize_every must be >= 1")


@dataclass
class VectorTrainingResult(TrainingResult):
    episode_epsilons: list[float]
    env_steps: int
    optimizer_updates: int


@dataclass
class VectorReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor


class VectorReplayMemory:
    """Compact replay storage for vectorized flat-observation environments."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.states = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_states = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        count = len(actions)
        if count == 0:
            return

        if count > self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            next_states = next_states[-self.capacity :]
            rewards = rewards[-self.capacity :]
            terminated = terminated[-self.capacity :]
            count = self.capacity

        indices = (np.arange(count) + self.position) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.next_states[indices] = next_states
        self.rewards[indices] = rewards
        self.terminated[indices] = terminated

        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int, device: torch.device) -> VectorReplayBatch:
        indices = self.rng.integers(self.size, size=batch_size)

        return VectorReplayBatch(
            states=torch.as_tensor(self.states[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device).unsqueeze(1),
            next_states=torch.as_tensor(self.next_states[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


class VectorTrainer:
    """DQN trainer that batches many environments to feed small models faster.

    In LunarLander measurements this trainer was about 3x faster on L4 GPU than
    the single-environment Trainer/TunedTrainer while reaching similar returns.
    The trick is simple: collect transitions from many environments at once, store
    them in array batches, and run larger action/training batches so the GPU spends
    more time computing and less time handling tiny operations.

    The name "VectorTrainer" follows the Gymnasium VectorEnv terminology.
    """

    def __init__(
        self,
        env,
        seed: int | None = None,
        device=None,
        replay_memory_capacity: int = 100_000,
        model_factory: ModelFactory = DQN,
    ) -> None:
        self.env = env
        self.steps_done = 0
        self.optimizer_updates = 0
        self.epsilons: list[float] = []
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            set_vector_seeds(env, seed)

        observations, _ = env.reset(seed=seed)
        self.num_envs = int(env.num_envs)
        self.observation_shape = tuple(observations.shape[1:])
        n_observations = math.prod(self.observation_shape)
        n_actions = env.single_action_space.n

        self.q_net = model_factory(n_observations, n_actions).to(self.device)
        self.target_net = model_factory(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=0.0,
            weight_decay=0.01,
            amsgrad=True,
        )
        self.memory = VectorReplayMemory(
            replay_memory_capacity,
            self.observation_shape,
            seed=seed,
        )

    def train(self, config: VectorTrainingConfig, plotter=None) -> VectorTrainingResult:
        """Train q_net with batched experience from a vector environment."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = config.learning_rate

        observations, _ = self.env.reset()
        awaiting_reset = np.zeros(self.num_envs, dtype=np.bool_)
        running_returns = np.zeros(self.num_envs, dtype=np.float32)
        running_lengths = np.zeros(self.num_envs, dtype=np.int64)
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        episode_epsilons: list[float] = []

        while len(episode_returns) < config.num_episodes:
            states = observations
            actions = self._select_actions(states, config)
            (
                next_observations,
                rewards,
                terminated,
                truncated,
                _,
            ) = self.env.step(actions)

            stepped = ~awaiting_reset
            self.memory.push_batch(
                states[stepped],
                actions[stepped],
                next_observations[stepped],
                rewards[stepped],
                terminated[stepped],
            )

            previous_steps = self.steps_done
            self.steps_done += int(stepped.sum())

            running_returns[stepped] += rewards[stepped]
            running_lengths[stepped] += 1

            done = (terminated | truncated) & stepped
            episode_count_before_step = len(episode_returns)
            for env_index in np.flatnonzero(done):
                if len(episode_returns) >= config.num_episodes:
                    break
                episode_returns.append(float(running_returns[env_index]))
                episode_lengths.append(int(running_lengths[env_index]))
                running_returns[env_index] = 0.0
                running_lengths[env_index] = 0

            if len(episode_returns) > episode_count_before_step:
                new_episode_count = len(episode_returns) - episode_count_before_step
                new_epsilons = [self._exploration_rate(config)] * new_episode_count
                episode_epsilons.extend(new_epsilons)
                self.epsilons.extend(new_epsilons)
                self._after_episode(
                    episode_returns,
                    episode_lengths,
                    episode_epsilons,
                    config,
                    plotter,
                )

            self._optimize_due(config, previous_steps)
            observations = next_observations
            awaiting_reset = done

        return VectorTrainingResult(
            self.q_net,
            episode_returns,
            episode_lengths,
            episode_epsilons,
            self.steps_done,
            self.optimizer_updates,
        )

    def _after_episode(
        self,
        episode_returns: list[float],
        episode_lengths: list[int],
        episode_epsilons: list[float],
        config: VectorTrainingConfig,
        plotter=None,
    ) -> None:
        """Hook used by train() after one or more episodes have finished."""
        if plotter is not None:
            plotter.plot_returns(episode_returns, epsilons=episode_epsilons)

    def _select_actions(
        self,
        observations: np.ndarray,
        config: VectorTrainingConfig,
    ) -> np.ndarray:
        actions = self.env.action_space.sample()
        greedy = self.rng.random(self.num_envs) > self._exploration_rate(config)

        if greedy.any():
            states = torch.as_tensor(
                observations[greedy],
                dtype=torch.float32,
                device=self.device,
            ).flatten(start_dim=1)
            with torch.no_grad():
                actions[greedy] = self.q_net(states).argmax(1).cpu().numpy()

        return actions

    def _exploration_rate(
        self,
        config: VectorTrainingConfig,
        step: int | None = None,
    ) -> float:
        step = self.steps_done if step is None else step
        return config.eps_end + (
            config.eps_start - config.eps_end
        ) * math.exp(-1.0 * step / config.eps_decay)

    def _optimize_due(
        self,
        config: VectorTrainingConfig,
        previous_steps: int,
    ) -> None:
        if len(self.memory) < config.batch_size or self.steps_done < config.learning_starts:
            return

        first_step = max(previous_steps + 1, config.learning_starts, 1)
        remainder = first_step % config.optimize_every
        if remainder:
            first_step += config.optimize_every - remainder

        if first_step > self.steps_done:
            return

        updates = ((self.steps_done - first_step) // config.optimize_every) + 1
        for _ in range(updates):
            self._optimize_model(config)
            self._soft_target_update(config.tau)

    def _optimize_model(self, config: VectorTrainingConfig) -> None:
        batch = self.memory.sample(config.batch_size, self.device)

        q_values = self.q_net(batch.states.flatten(start_dim=1)).gather(1, batch.actions)

        with torch.no_grad():
            next_q_values = self.target_net(
                batch.next_states.flatten(start_dim=1),
            ).max(1).values
            next_q_values[batch.terminated] = 0.0
            td_targets = batch.rewards + config.gamma * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values.squeeze(1), td_targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()
        self.optimizer_updates += 1

    def _soft_target_update(self, tau: float) -> None:
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.q_net.parameters(),
                strict=True,
            ):
                target_param.lerp_(policy_param, tau)


def set_vector_seeds(env, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    env.action_space.seed(seed)
    env.single_action_space.seed(seed)
    env.single_observation_space.seed(seed)
