"""Logging helpers for LunarLander HPO."""

import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any


def configure_file_logging(
    study_dir: str | Path,
    filename: str = "hpo.log",
) -> None:
    log_path = Path(study_dir) / filename
    logger = logging.getLogger("hpo")
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        return

    handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_call(func: Callable[..., Any]) -> Callable[..., Any]:
    call_logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_logger.info("start %s", func.__name__)
        try:
            return func(*args, **kwargs)
        finally:
            call_logger.info("finish %s", func.__name__)

    return wrapper
