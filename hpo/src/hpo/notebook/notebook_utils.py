"""Small helpers shared by HPO notebooks."""

from collections.abc import Sequence
from typing import Any


def neighbors(value: Any, choices: Sequence[Any]) -> list[Any]:
    """Return value plus its direct neighbors in choices."""
    index = choices.index(value)
    return list(choices[max(0, index - 1):index + 2])
