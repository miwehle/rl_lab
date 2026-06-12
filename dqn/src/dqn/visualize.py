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
        self.episode_marks: dict[str, list[int]] = {}
        self.epsilons: list[float] | None = None
        self.is_ipython = "inline" in matplotlib.get_backend()
        self.display: Any = None
        self.figure: Any = None
        self.ax_returns: Any = None
        self.ax_epsilon: Any = None
        self.returns_line: Any = None
        self.mean_line: Any = None
        self.epsilon_line: Any = None
        self.mark_artists: list[Any] = []

        if self.is_ipython:
            from IPython import display

            self.display = display

        plt.ion()

    def mark_episode(self, episode: int, label: str, repeat: bool = False) -> None:
        episodes = self.episode_marks.setdefault(label, [])
        if repeat or not episodes:
            episodes.append(episode)

    def plot_returns(
        self,
        returns: list[float],
        show_result: bool = False,
        epsilons: list[float] | None = None,
    ) -> None:
        self._ensure_plot()
        ax_returns = self.ax_returns
        ax_returns.set_title("Result" if show_result else "Training...")
        episodes = list(range(len(returns)))
        self.returns_line.set_data(episodes, returns)

        rolling_window = 50
        if len(returns) >= rolling_window:
            returns_t = torch.tensor(returns, dtype=torch.float)
            means = returns_t.unfold(0, rolling_window, 1).mean(1).view(-1)
            mean_episodes = list(range(rolling_window - 1, rolling_window - 1 + len(means)))
            self.mean_line.set_data(mean_episodes, means.numpy())
            self.mean_line.set_visible(True)
        else:
            self.mean_line.set_data([], [])
            self.mean_line.set_visible(False)

        legend = ax_returns.get_legend()
        if legend is not None:
            legend.remove()

        for artist in self.mark_artists:
            artist.remove()
        self.mark_artists.clear()
        for label, episodes in self.episode_marks.items():
            color = "gray" if label == "Checkpoint" else "tab:red"
            for index, episode in enumerate(episodes):
                legend_label = label if index == 0 else "_nolegend_"
                artist = ax_returns.axvline(
                    episode,
                    color=color,
                    linestyle=":",
                    label=legend_label,
                )
                self.mark_artists.append(artist)

        if self.episode_marks:
            ax_returns.legend(loc="lower right")

        if epsilons is not None:
            self.epsilons = list(epsilons)
        elif self.epsilons is not None and len(self.epsilons) != len(returns):
            self.epsilons = None

        epsilons = self.epsilons

        if epsilons is not None:
            self.ax_epsilon.set_visible(True)
            self.epsilon_line.set_data(list(range(len(epsilons))), epsilons)
            self.epsilon_line.set_visible(True)
        else:
            self.epsilon_line.set_data([], [])
            self.epsilon_line.set_visible(False)
            self.ax_epsilon.set_visible(False)

        ax_returns.relim()
        ax_returns.autoscale_view()
        self.figure.canvas.draw_idle()

        plt.pause(0.001)

        if self.display is not None:
            self.display.display(self.figure)
            if not show_result:
                self.display.clear_output(wait=True)

    def _ensure_plot(self) -> None:
        if self.figure is not None:
            return

        self.figure, self.ax_returns = plt.subplots()
        self.ax_returns.set_xlabel("Episode")
        self.ax_returns.set_ylabel(self.y_label)
        (self.returns_line,) = self.ax_returns.plot([], [])
        (self.mean_line,) = self.ax_returns.plot([], [])

        self.ax_epsilon = self.ax_returns.twinx()
        self.ax_epsilon.set_ylabel("Epsilon")
        self.ax_epsilon.set_ylim(0, 1)
        (self.epsilon_line,) = self.ax_epsilon.plot(
            [],
            [],
            color="tab:green",
            linestyle="--",
        )
        self.ax_epsilon.set_visible(False)


def record_episode(
    make_env: Callable[[], Any],
    q_net: DQN,
    device: torch.device,
    max_steps: int = 500,
    return_scores: bool = False,
) -> list[Any] | tuple[list[Any], list[float]]:
    env = make_env()
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    frames = []
    scores = []
    episode_return = 0.0
    q_net.eval()

    for _ in range(max_steps):
        frames.append(env.render())
        scores.append(episode_return)

        with torch.no_grad():
            action = q_net(state).argmax(dim=1).item()

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)

        if terminated or truncated:
            frames.append(env.render())
            scores.append(episode_return)
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()
    if return_scores:
        return frames, scores
    return frames


def show_animation(
    frames: list[Any],
    interval: int = 20,
    scores: list[float] | None = None,
) -> Any:
    fig = plt.figure()
    plt.axis("off")
    image = plt.imshow(frames[0])
    score_text = None

    if scores is not None:
        score_text = plt.text(
            0.02,
            0.95,
            "",
            color="white",
            fontsize=12,
            transform=plt.gca().transAxes,
        )

    animation_frames = list(enumerate(frames))

    def update(frame_data: tuple[int, Any]) -> list[Any]:
        frame_index, frame = frame_data
        image.set_array(frame)
        artists = [image]

        if score_text is not None:
            score = scores[frame_index]
            score_text.set_text(f"Return: {score:.0f}")
            artists.append(score_text)

        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=animation_frames,
        interval=interval,
        blit=True,
    )
    plt.close(fig)

    try:
        from IPython.display import HTML

        return HTML(ani.to_jshtml())
    except ImportError:
        return ani
