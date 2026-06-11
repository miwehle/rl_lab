"""Plotting and rendering helpers for DQN training results."""

from collections.abc import Callable
from typing import Any

import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import torch

from dqn.model import DQN


class EpisodePlotter:
    def __init__(self, y_label: str = "Return") -> None:
        self.y_label = y_label
        self.episode_marks: dict[str, int] = {}
        self.epsilons: list[float] | None = None
        self.is_ipython = "inline" in matplotlib.get_backend()
        self.display: Any = None

        if self.is_ipython:
            from IPython import display

            self.display = display

        plt.ion()

    def mark_episode(self, episode: int, label: str) -> None:
        self.episode_marks.setdefault(label, episode)

    def plot_returns(
        self,
        returns: list[float],
        show_result: bool = False,
        epsilons: list[float] | None = None,
    ) -> None:
        plt.figure(1)
        plt.clf()
        ax_returns = plt.gca()
        ax_returns.set_title("Result" if show_result else "Training...")
        ax_returns.set_xlabel("Episode")
        ax_returns.set_ylabel(self.y_label)
        ax_returns.plot(returns)

        rolling_window = 50
        if len(returns) >= rolling_window:
            returns_t = torch.tensor(returns, dtype=torch.float)
            means = returns_t.unfold(0, rolling_window, 1).mean(1).view(-1)
            mean_episodes = range(rolling_window - 1, rolling_window - 1 + len(means))
            ax_returns.plot(mean_episodes, means.numpy())

        for label, episode in self.episode_marks.items():
            ax_returns.axvline(episode, color="tab:red", linestyle=":", label=label)

        if self.episode_marks:
            ax_returns.legend(loc="lower right")

        if epsilons is not None:
            self.epsilons = list(epsilons)
        elif self.epsilons is not None and len(self.epsilons) != len(returns):
            self.epsilons = None

        epsilons = self.epsilons

        if epsilons is not None:
            ax_epsilon = ax_returns.twinx()
            ax_epsilon.set_ylabel("Epsilon")
            ax_epsilon.set_ylim(0, 1)
            ax_epsilon.plot(epsilons, color="tab:green", linestyle="--")

        plt.pause(0.001)

        if self.display is not None:
            self.display.display(plt.gcf())
            if not show_result:
                self.display.clear_output(wait=True)


def record_episode(
    make_env: Callable[[], Any],
    q_net: DQN,
    device: torch.device,
    max_steps: int = 500,
) -> list[Any]:
    env = make_env()
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    frames = []
    q_net.eval()

    for _ in range(max_steps):
        frames.append(env.render())

        with torch.no_grad():
            action = q_net(state).argmax(dim=1).item()

        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()
    return frames


def show_animation(frames: list[Any], interval: int = 20) -> Any:
    fig = plt.figure()
    plt.axis("off")
    image = plt.imshow(frames[0])

    def update(frame: Any) -> list[Any]:
        image.set_array(frame)
        return [image]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close(fig)

    try:
        from IPython.display import HTML

        return HTML(ani.to_jshtml())
    except ImportError:
        return ani
