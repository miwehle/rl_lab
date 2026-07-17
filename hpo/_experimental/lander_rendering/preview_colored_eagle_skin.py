"""Render local debug previews for the colored Eagle lander skin."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "hpo" / "src"))
sys.path.insert(0, str(REPO_ROOT / "dqn" / "src"))

import pygame  # noqa: E402

from hpo.environments.solar_system_lander.env import DEFAULT_WORLD_MIX, EnvFactory, World  # noqa: E402
from hpo.evaluation.rendering.solar_system_lander._colors import LanderOverlay, world_colors  # noqa: E402
from hpo.evaluation.rendering.solar_system_lander._scene import LanderRenderWrapper  # noqa: E402
from hpo.evaluation.rendering.solar_system_lander._skins.colored_eagle import (  # noqa: E402
    ColoredEagleSkin,
    _assets,
)


DEFAULT_OUT_DIR = Path(r"C:\tmp\eagle_skin_preview") if os.name == "nt" else Path("/tmp/eagle_skin_preview")
DEFAULT_INKSCAPE = Path(
    os.environ.get(
        "INKSCAPE_EXE",
        r"C:\Apps\inkscape\bin\inkscape.exe" if os.name == "nt" else "inkscape",
    )
)
ASSET_DIR = REPO_ROOT / "hpo" / "src" / "hpo" / "evaluation" / "rendering" / "solar_system_lander" / "_skin_assets" / "eagle_colored"
SVG_TO_PNG = (
    (ASSET_DIR / "eagle_colored_body.svg", ASSET_DIR / "eagle_colored_body.png"),
    (ASSET_DIR / "eagle_colored_side_legs.svg", ASSET_DIR / "eagle_colored_side_legs.png"),
)
SKY = (143, 199, 232)
CHECKER_A = (214, 219, 224)
CHECKER_B = (245, 247, 249)


def main() -> None:
    args = _parse_args()
    out_dir = args.out.resolve()
    if args.clean:
        _clean_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_assets:
        _refresh_assets(args.inkscape)

        pygame.init()
    try:
        skin = ColoredEagleSkin(scale=args.scale)
        body, side_legs = _assets()
        lines = _asset_summary(skin, body, side_legs)

        _save_asset_previews(out_dir, skin, body.surface, "body")
        _save_asset_previews(out_dir, skin, side_legs.surface, "side_legs")

        frame_paths = _save_lander_frames(
            out_dir,
            skin=skin,
            world=World(args.world),
            seed=args.seed,
            steps=args.steps,
            overlay=args.overlay,
        )
        upright_paths = _save_upright_pose(out_dir, skin=skin, world=World(args.world), seed=args.seed)

        summary_path = out_dir / "summary.txt"
        summary_path.write_text(
            "\n".join(
                lines
                + ["", "Frames:"]
                + [str(path) for path in frame_paths]
                + ["", "Upright:"]
                + [str(path) for path in upright_paths]
            )
            + "\n"
        )
    finally:
        pygame.quit()

    print(f"Wrote colored Eagle preview to: {out_dir}")
    print(summary_path.read_text())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scale", type=float, default=ColoredEagleSkin().scale)
    parser.add_argument("--world", choices=[world.value for world in World], default=World.EARTH.value)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 15, 45])
    parser.add_argument("--refresh-assets", action="store_true")
    parser.add_argument("--inkscape", type=Path, default=DEFAULT_INKSCAPE)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false")
    parser.set_defaults(overlay=True)
    return parser.parse_args()


def _refresh_assets(inkscape: Path) -> None:
    for svg_path, png_path in SVG_TO_PNG:
        subprocess.run(
            [
                str(inkscape),
                "--export-type=png",
                f"--export-filename={png_path}",
                str(svg_path),
            ],
            check=True,
        )


def _clean_out_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _asset_summary(skin, body, side_legs) -> list[str]:
    return [
        f"skin scale: {skin.scale}",
        f"body loaded: {body.surface.get_size()}, anchor: {body.anchor}",
        f"side legs loaded: {side_legs.surface.get_size()}, anchor: {side_legs.anchor}",
        f"body target: {_scaled_size(body.surface, skin.scale)}",
        f"side legs target: {_scaled_size(side_legs.surface, skin.scale)}",
    ]


def _save_asset_previews(out_dir: Path, skin: ColoredEagleSkin, surface, name: str) -> None:
    _save_on_background(surface, out_dir / f"{name}_loaded_checker.png", checker=True)
    _save_on_background(surface, out_dir / f"{name}_loaded_sky.png", color=SKY)

    scaled = pygame.transform.smoothscale(surface, _scaled_size(surface, skin.scale))
    _save_on_background(scaled, out_dir / f"{name}_target.png", color=SKY)

    pixel_preview = pygame.transform.scale(scaled, (scaled.get_width() * 4, scaled.get_height() * 4))
    _save_on_background(pixel_preview, out_dir / f"{name}_target_4x.png", color=SKY)


def _save_lander_frames(
    out_dir: Path, *, skin: ColoredEagleSkin, world: World, seed: int, steps: list[int], overlay: bool
) -> list[Path]:
    factory = EnvFactory("10d", world_mix=DEFAULT_WORLD_MIX)
    env = LanderRenderWrapper(
        factory.make_env(world, render_mode="rgb_array"),
        colors=world_colors([world])[0],
        overlay=LanderOverlay() if overlay else None,
        skin=skin,
    )
    paths = []
    try:
        env.reset(seed=seed)
        current_step = 0
        for target_step in sorted(set(steps)):
            while current_step < target_step:
                env.step(0)
                current_step += 1
            frame = env.render()
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            path = out_dir / f"frame_{world.value}_seed{seed}_step{target_step:03d}.png"
            pygame.image.save(surface, path)
            paths.append(path)
    finally:
        env.close()
    return paths


def _save_upright_pose(out_dir: Path, *, skin: ColoredEagleSkin, world: World, seed: int) -> list[Path]:
    factory = EnvFactory("10d", world_mix=DEFAULT_WORLD_MIX)
    env = factory.make_env(world, render_mode="rgb_array")
    try:
        env.reset(seed=seed)
        lander = env.unwrapped.lander
        right_leg, left_leg = env.unwrapped.legs
        right_rel = (right_leg.position.x - lander.position.x, right_leg.position.y - lander.position.y)
        left_rel = (left_leg.position.x - lander.position.x, left_leg.position.y - lander.position.y)
    finally:
        env.close()

    body_position = (10.0, 6.7)
    fake_env = SimpleNamespace(
        lander=SimpleNamespace(position=body_position, angle=0.0),
        legs=[
            SimpleNamespace(
                position=(body_position[0] + right_rel[0], body_position[1] + right_rel[1]),
                angle=skin.right_leg_rest_angle,
            ),
            SimpleNamespace(
                position=(body_position[0] + left_rel[0], body_position[1] + left_rel[1]),
                angle=skin.left_leg_rest_angle,
            ),
        ],
    )

    surface = pygame.Surface((600, 400))
    surface.fill(SKY)
    skin.draw(surface, fake_env)

    upright_path = out_dir / "upright_pose.png"
    pygame.image.save(surface, upright_path)

    crop = pygame.Surface((150, 150))
    crop.blit(surface, (0, 0), (225, 105, 150, 150))
    crop = pygame.transform.scale(crop, (crop.get_width() * 4, crop.get_height() * 4))
    crop_path = out_dir / "upright_pose_4x.png"
    pygame.image.save(crop, crop_path)
    return [upright_path, crop_path]


def _scaled_size(surface, scale: float) -> tuple[int, int]:
    return max(1, round(surface.get_width() * scale)), max(1, round(surface.get_height() * scale))


def _save_on_background(surface, path: Path, *, color=None, checker: bool = False) -> None:
    target = pygame.Surface(surface.get_size())
    if checker:
        _draw_checker(target)
    else:
        target.fill(color or SKY)
    target.blit(surface, (0, 0))
    pygame.image.save(target, path)


def _draw_checker(surface) -> None:
    tile = 12
    for y in range(0, surface.get_height(), tile):
        for x in range(0, surface.get_width(), tile):
            color = CHECKER_A if (x // tile + y // tile) % 2 == 0 else CHECKER_B
            pygame.draw.rect(surface, color, (x, y, tile, tile))


if __name__ == "__main__":
    main()
