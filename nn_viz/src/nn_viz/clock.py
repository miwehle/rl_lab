from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from time import perf_counter

type LapLabel = str | None

"""Lightweight timing utilities for bottleneck analysis.

All durations are measured in seconds.
Unlabeled laps are aggregated under None.
"""


def _now() -> float:
    return perf_counter()


class Clock:
    def __init__(self, name: str, *, now: float):
        self.name = name
        self._started_at = now
        self._last_lap_at = now
        self._stopped_at: float | None = None
        self._lap_times: dict[LapLabel, float] = defaultdict(float)
        self.reused = False

    @property
    def stopped(self) -> bool:
        return self._stopped_at is not None

    @property
    def total_time(self) -> float | None:
        if self._stopped_at is None:
            return None
        return self._stopped_at - self._started_at

    @property
    def lap_times(self) -> dict[LapLabel, float]:
        return dict(self._lap_times)


class _ClockRegistry:
    def __init__(self, time_source: Callable[[], float]):
        self._time_source = time_source
        self._clocks_by_name: dict[str, list[Clock]] = defaultdict(list)
        self._running_clocks: dict[str, Clock] = {}

    def get_clock(self, name: str) -> Clock:
        running_clock = self._running_clocks.get(name)
        if running_clock is not None:
            running_clock.reused = True
            return running_clock
        clock = Clock(name, now=self._time_source())
        self._clocks_by_name[name].append(clock)
        self._running_clocks[name] = clock
        return clock

    def lap(self, clock: Clock, label: LapLabel = None) -> float:
        if clock.stopped:
            return 0.0
        now = self._time_source()
        duration = now - clock._last_lap_at
        clock._lap_times[label] += duration
        clock._last_lap_at = now
        return duration

    def stop(self, clock: Clock, label: LapLabel = None) -> float | None:
        if clock.stopped:
            return clock.total_time
        now = self._time_source()
        duration = now - clock._last_lap_at
        clock._lap_times[label] += duration
        clock._last_lap_at = now
        clock._stopped_at = now
        self._running_clocks.pop(clock.name, None)
        return clock.total_time

    def total_lap_times(self, name: str) -> dict[LapLabel, float]:
        totals: dict[LapLabel, float] = defaultdict(float)
        for clock in self._clocks_by_name.get(name, []):
            for label, duration in clock._lap_times.items():
                totals[label] += duration
        return dict(totals)

    def total_time(self, name: str) -> float:
        return sum(clock.total_time or 0.0 for clock in self._clocks_by_name.get(name, []))

    def reset(self) -> None:
        self._clocks_by_name.clear()
        self._running_clocks.clear()

    def is_in_use(self, name: str) -> bool:
        return name in self._running_clocks


_registry = _ClockRegistry(_now)


def get_clock(name: str) -> Clock:
    return _registry.get_clock(name)


def lap(clock: Clock, label: LapLabel = None) -> float:
    return _registry.lap(clock, label)


def stop(clock: Clock, label: LapLabel = None) -> float | None:
    return _registry.stop(clock, label)


def total_lap_times(name: str) -> dict[LapLabel, float]:
    return _registry.total_lap_times(name)


def total_time(name: str) -> float:
    return _registry.total_time(name)


def reset_clocks() -> None:
    _registry.reset()


def is_in_use(name: str) -> bool:
    """Return whether a clock with this name is currently running."""
    return _registry.is_in_use(name)
