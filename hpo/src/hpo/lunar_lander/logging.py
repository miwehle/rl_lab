"""Logging helpers for LunarLander HPO."""

import logging
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any


_call_depth: ContextVar[int] = ContextVar("hpo_call_depth", default=0)


class _SourceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        line = getattr(record, "definition_line", record.lineno)
        source = f"{record.name}:{line}"
        record.source = source if len(source) <= 32 else f"…{source[-31:]}"
        depth = _call_depth.get()
        record.indent = " " * (depth if getattr(record, "call_boundary", False) else depth + 2)
        return True


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
    handler.addFilter(_SourceFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(source)-32s %(indent)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_call(func: Callable[..., Any]) -> Callable[..., Any]:
    call_logger = logging.getLogger(func.__module__)
    log_extra = {
        "definition_line": func.__code__.co_firstlineno,
        "call_boundary": True,
    }

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_logger.info("-> %s", func.__name__, extra=log_extra)
        token = _call_depth.set(_call_depth.get() + 1)
        try:
            return func(*args, **kwargs)
        finally:
            _call_depth.reset(token)
            call_logger.info("<- %s", func.__name__, extra=log_extra)

    return wrapper
