"""Shared dashboard style constants and helpers."""

from typing import Any

NO_DATA_TEXT = "No data"
EMPTY_SCORE_RANGE = [0, 250]


def set_empty_score_yaxis(
    figure: Any, *, row: int, col: int, title_text: str = "Score", secondary_y: bool = False
) -> None:
    figure.update_yaxes(
        title_text=title_text, range=EMPTY_SCORE_RANGE, row=row, col=col, secondary_y=secondary_y
    )


def hide_empty_xaxis(figure: Any, *, row: int, col: int) -> None:
    figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
