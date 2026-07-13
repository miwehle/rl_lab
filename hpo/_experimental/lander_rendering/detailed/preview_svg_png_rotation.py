"""Preview CairoSVG-rendered Eagle PNGs rotated by Pygame.

This is an experimental quality check only. It does not change the production
renderer and deliberately keeps crop/scale parameters local to this script.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pygame
import numpy as np
from PIL import Image

try:
    import cairosvg
except (ImportError, OSError):  # pragma: no cover - developer convenience
    cairosvg = None


HERE = Path(__file__).resolve().parent
SVG_PATH = HERE / "eagle.svg"
OUTPUT_PATH = HERE / "eagle_svg_png_rotation_preview.png"
SVG_ASSET_PATH = HERE / "eagle_svg_raster_asset.png"
SKY = (143, 199, 232)
PYGAME_DIR = HERE

if str(PYGAME_DIR) not in sys.path:
    sys.path.insert(0, str(PYGAME_DIR))

from eagle_lander_pygame import EAGLE_OPS, SVG_HEIGHT, SVG_WIDTH, make_eagle_surface  # noqa: E402


def main() -> None:
    args = _parse_args()
    pygame.init()
    pygame.font.init()
    try:
        svg_asset = _render_svg_asset(args)
        pygame_asset = make_eagle_surface(scale=args.display_scale, background=None)
        svg_asset.save(SVG_ASSET_PATH)
        preview = _make_preview(svg_asset, pygame_asset, args)
        pygame.image.save(preview, args.output)
        print(args.output)
        print(SVG_ASSET_PATH)
    finally:
        pygame.quit()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", type=Path, default=SVG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--render-scale", type=float, default=None)
    parser.add_argument("--display-scale", type=float, default=0.16)
    parser.add_argument("--preview-zoom", type=int, default=4)
    parser.add_argument("--angles", type=float, nargs="+", default=[-35.0, 0.0, 35.0])
    parser.add_argument("--backend", choices=["auto", "cairosvg", "browser"], default="auto")
    parser.add_argument(
        "--source",
        choices=["ops", "svg"],
        default="ops",
        help="ops renders the cleaned Eagle ops as SVG; svg renders eagle.svg directly.",
    )
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Optional crop in SVG viewBox units after CairoSVG rendering.",
    )
    return parser.parse_args()


def _render_svg_asset(args: argparse.Namespace) -> Image.Image:
    with tempfile.TemporaryDirectory() as temp_dir:
        svg = _source_svg(args, Path(temp_dir))
        image = _render_with_cairosvg(args, svg) if args.backend != "browser" else None
        if image is None:
            image = _render_with_browser(args, svg)
        if args.crop is None:
            return image if image.getbbox() is None else image.crop(image.getbbox())

        x, y, width, height = args.crop
        scale = _render_scale(args)
        crop = (
            round(x * scale),
            round(y * scale),
            round((x + width) * scale),
            round((y + height) * scale),
        )
        return image.crop(crop)


def _source_svg(args: argparse.Namespace, temp_dir: Path) -> Path:
    if args.source == "svg":
        return args.svg
    path = temp_dir / "eagle_ops.svg"
    path.write_text(_ops_svg(), encoding="utf-8")
    return path


def _ops_svg() -> str:
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
    ]
    for kind, fill, stroke, stroke_width, points in EAGLE_OPS:
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        style = _svg_style(fill, stroke, stroke_width)
        if kind == "polygon":
            items.append(f'<polygon points="{point_text}" {style}/>')
        elif kind == "polyline":
            items.append(f'<polyline points="{point_text}" {style}/>')
    items.append("</svg>")
    return "\n".join(items)


def _svg_style(fill: str | None, stroke: str | None, stroke_width: float) -> str:
    fill = "none" if fill is None else fill
    stroke = "none" if stroke is None else stroke
    return f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"'


def _render_with_cairosvg(args: argparse.Namespace, svg: Path) -> Image.Image | None:
    if cairosvg is None:
        if args.backend == "cairosvg":
            raise SystemExit("Install CairoSVG first: python -m pip install cairosvg")
        return None
    try:
        png = cairosvg.svg2png(url=str(svg), scale=_render_scale(args))
    except OSError as exc:
        if args.backend == "cairosvg":
            raise
        print(f"CairoSVG unavailable ({exc}); falling back to browser screenshot.")
        return None
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _render_with_browser(args: argparse.Namespace, svg: Path) -> Image.Image:
    black = _browser_screenshot(args, svg, (0, 0, 0))
    white = _browser_screenshot(args, svg, (255, 255, 255))
    return _transparent_from_black_white(black, white)


def _browser_screenshot(args: argparse.Namespace, svg: Path, background: tuple[int, int, int]) -> Image.Image:
    browser = _browser_path()
    if browser is None:
        raise SystemExit("No CairoSVG backend and no Chrome/Edge executable found.")

    width, height = _svg_dimensions(args)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        html = temp / "preview.html"
        screenshot = temp / "preview.png"
        html.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    f"<html><body style='margin:0;background:rgb{background};overflow:hidden'>",
                    f"<img src='{svg.resolve().as_uri()}' style='width:{width}px;height:{height}px;display:block'>",
                    "</body></html>",
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                f"--screenshot={screenshot}",
                f"--window-size={width},{height}",
                html.resolve().as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return Image.open(screenshot).convert("RGBA")


def _transparent_from_black_white(black: Image.Image, white: Image.Image) -> Image.Image:
    black_rgb = np.asarray(black.convert("RGB"), dtype=np.float32)
    white_rgb = np.asarray(white.convert("RGB"), dtype=np.float32)
    alpha = 255.0 - np.max(white_rgb - black_rgb, axis=2)
    safe_alpha = np.where(alpha <= 0.0, 1.0, alpha)
    rgb = np.clip(black_rgb * 255.0 / safe_alpha[:, :, None], 0, 255)
    rgba = np.dstack((rgb, alpha)).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _svg_dimensions(args: argparse.Namespace) -> tuple[int, int]:
    scale = _render_scale(args)
    if args.source == "ops":
        return round(SVG_WIDTH * scale), round(SVG_HEIGHT * scale)
    return round(535 * scale), round(989 * scale)


def _render_scale(args: argparse.Namespace) -> float:
    return args.display_scale if args.render_scale is None else args.render_scale


def _browser_path() -> Path | None:
    for candidate in (
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _make_preview(svg_asset: Image.Image, pygame_asset: pygame.Surface, args: argparse.Namespace) -> pygame.Surface:
    svg_surface = _pil_to_surface(svg_asset)
    target_width = pygame_asset.get_width()
    target_height = pygame_asset.get_height()
    if svg_surface.get_size() != (target_width, target_height):
        svg_surface = pygame.transform.smoothscale(svg_surface, (target_width, target_height))

    font = pygame.font.Font(None, 24)
    labels = ("Pygame ops", "SVG PNG + Pygame rotate")
    rows = [
        [_rotate(surface, angle) for angle in args.angles]
        for surface in (pygame_asset, svg_surface)
    ]
    cell_width = max(item.get_width() for row in rows for item in row) * args.preview_zoom + 40
    cell_height = max(item.get_height() for row in rows for item in row) * args.preview_zoom + 62
    sheet = pygame.Surface((cell_width * len(args.angles), cell_height * 2), pygame.SRCALPHA)
    sheet.fill((*SKY, 255))

    for row_index, row in enumerate(rows):
        label = font.render(labels[row_index], True, (20, 20, 20))
        sheet.blit(label, (12, row_index * cell_height + 8))
        for column_index, rotated in enumerate(row):
            zoomed = pygame.transform.scale(
                rotated,
                (rotated.get_width() * args.preview_zoom, rotated.get_height() * args.preview_zoom),
            )
            x = column_index * cell_width + (cell_width - zoomed.get_width()) // 2
            y = row_index * cell_height + 44 + (cell_height - 54 - zoomed.get_height()) // 2
            sheet.blit(zoomed, (x, y))
            angle_label = font.render(f"{args.angles[column_index]:+.0f} deg", True, (20, 20, 20))
            sheet.blit(angle_label, (column_index * cell_width + 12, row_index * cell_height + cell_height - 26))
    return sheet


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    return pygame.image.frombuffer(image.tobytes(), image.size, "RGBA")


def _rotate(surface: pygame.Surface, angle: float) -> pygame.Surface:
    return pygame.transform.rotozoom(surface, angle, 1.0)


if __name__ == "__main__":
    main()
