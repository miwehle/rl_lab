"""Play the SolarSystemLander environments with cursor keys."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from gymnasium.envs.box2d import lunar_lander

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "dqn" / "src"))
sys.path.insert(0, str(ROOT / "hpo" / "src"))

from hpo.evaluation.lander_rendering import (  # noqa: E402
    LanderOverlay,
    LanderRenderWrapper,
    world_colors,
)
from hpo.solar_system_lander.environment import EnvFactory, World  # noqa: E402

WORLDS = [world.value for world in World]
NORMAL_PERIOD = 3
BOOST_PERIOD = 1
WINDOW_TITLE = (
    "SolarSystemLander | arrows thrust | space boost | R restart | N new seed | "
    "1-5 world | Esc quit"
)


@dataclass
class GameState:
    world_index: int
    seed: int
    step: int = 0
    score: float = 0.0
    done: bool = False
    status: str = "flying"


def main() -> None:
    args = _parse_args()

    import pygame

    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((lunar_lander.VIEWPORT_W, lunar_lander.VIEWPORT_H))
    clock = pygame.time.Clock()

    factory = EnvFactory(args.observation_mode)
    colors = dict(zip(WORLDS, world_colors(WORLDS), strict=True))
    state = GameState(world_index=WORLDS.index(args.world), seed=args.seed)
    env = _make_env(factory, state.world_index, colors)
    env.reset(seed=state.seed)

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.close()
                        state = GameState(
                            world_index=state.world_index,
                            seed=state.seed,
                        )
                        env = _make_env(factory, state.world_index, colors)
                        env.reset(seed=state.seed)
                    elif event.key == pygame.K_n:
                        env.close()
                        state = GameState(
                            world_index=state.world_index,
                            seed=state.seed + 1,
                        )
                        env = _make_env(factory, state.world_index, colors)
                        env.reset(seed=state.seed)
                    elif pygame.K_1 <= event.key <= pygame.K_5:
                        env.close()
                        state = GameState(
                            world_index=event.key - pygame.K_1,
                            seed=state.seed,
                        )
                        env = _make_env(factory, state.world_index, colors)
                        env.reset(seed=state.seed)

            if not state.done:
                action = _action_from_keys(pygame.key.get_pressed(), state.step)
                _, reward, terminated, truncated, _ = env.step(action)
                state.step += 1
                state.score += float(reward)
                if terminated or truncated:
                    state.done = True
                    state.status = "landed" if terminated else "truncated"

            frame = env.render()
            if frame is not None:
                _draw_frame(screen, frame)
                _draw_hud(screen, state)
                pygame.display.flip()

            clock.tick(lunar_lander.FPS)
    finally:
        env.close()
        pygame.quit()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=WORLDS, default=World.EARTH.value)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--observation-mode",
        choices=["8d", "9d", "10d", "11d"],
        default="10d",
    )
    return parser.parse_args()


def _make_env(factory: EnvFactory, world_index: int, colors: dict[str, object]):
    world = WORLDS[world_index]
    env = factory.make_env(world, render_mode="rgb_array")
    return LanderRenderWrapper(
        env,
        colors=colors[world],
        overlay=LanderOverlay(),
    )


def _action_from_keys(keys, step: int) -> int:
    if not _should_fire(keys, step):
        return 0
    if keys[_key("up")]:
        return 2
    if keys[_key("left")] and not keys[_key("right")]:
        return 1
    if keys[_key("right")] and not keys[_key("left")]:
        return 3
    return 0


def _should_fire(keys, step: int) -> bool:
    period = BOOST_PERIOD if keys[_key("space")] else NORMAL_PERIOD
    return step % period == 0


def _key(name: str) -> int:
    import pygame

    return {
        "up": pygame.K_UP,
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "space": pygame.K_SPACE,
    }[name]


def _draw_frame(screen, frame: np.ndarray) -> None:
    import pygame

    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surface, (0, 0))


def _draw_hud(screen, state: GameState) -> None:
    import pygame

    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, 20)
    text = (
        f"{WORLDS[state.world_index].title()} | seed {state.seed} | "
        f"t {state.step / lunar_lander.FPS:.1f}s | score {state.score:.1f}"
    )
    if state.done:
        text = f"{text} | {state.status} | R restart"
    shadow = font.render(text, True, (0, 0, 0))
    label = font.render(text, True, (255, 255, 255))
    y = lunar_lander.VIEWPORT_H - 24
    screen.blit(shadow, (9, y + 1))
    screen.blit(label, (8, y))


if __name__ == "__main__":
    main()
