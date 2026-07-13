"""Render-facing view of SolarSystemLander Gym/env state."""

from dataclasses import dataclass
import math

from gymnasium.envs.box2d import lunar_lander
from gymnasium.utils import seeding


@dataclass(frozen=True)
class WindState:
    """Render-facing wind state prepared from Gym/env internals."""

    acceleration: float | None
    max_acceleration: float | None

    @classmethod
    def from_env(cls, env, mass) -> "WindState":
        if not getattr(env, "enable_wind", False):
            return cls(acceleration=None, max_acceleration=None)
        wind_idx = getattr(env, "wind_idx", None)
        if wind_idx is None or not mass:
            return cls(acceleration=None, max_acceleration=None)

        wind_power = float(getattr(env, "wind_power", 0.0))
        acceleration = _force_wave(wind_idx) * wind_power / float(mass)
        return cls(
            acceleration=acceleration,
            max_acceleration=_max_acceleration(wind_power, mass),
        )


@dataclass(frozen=True)
class EnvState:
    """Render-facing env state prepared once per frame from Gym/env internals."""

    world_name: str | None
    gravity: float | None
    score: float | None
    steps_since_reset: int
    wind: WindState
    turbulence_acceleration: float | None
    turbulence_max_acceleration: float | None
    weather_turbulence_max_degrees: float | None
    initial_kick_delta_v: float | None
    initial_kick_direction: tuple[float, float] | None
    lander_screen_position: tuple[int, int] | None

    @classmethod
    def from_env(cls, *, wrapper, env) -> "EnvState":
        """Extract render values from the LanderRenderWrapper and unwrapped Gym env.

        wrapper:
            The LanderRenderWrapper. Provides render-time bookkeeping such as
            reset seed, score, and steps since reset.

        env:
            The unwrapped Gym/Box2D env. Provides world and physics state such
            as lander body, legs, weather, wind, and turbulence.
        """
        env = getattr(env, "unwrapped", env)
        world = getattr(env, "world", None)
        lander = getattr(env, "lander", None)
        mass = getattr(lander, "mass", None)
        inertia = getattr(lander, "inertia", None)
        turbulence = _turbulence_acceleration(env)
        weather_turbulence_max_degrees = _weather_turbulence_max(env, inertia)
        kick = _initial_kick(getattr(wrapper, "reset_seed", None), mass)
        return cls(
            world_name=_world_name(world),
            gravity=_world_gravity(world),
            score=getattr(wrapper, "score", None),
            steps_since_reset=getattr(wrapper, "steps_since_reset", 0),
            wind=WindState.from_env(env, mass),
            turbulence_acceleration=None if turbulence is None else turbulence[0],
            turbulence_max_acceleration=None if turbulence is None else turbulence[1],
            weather_turbulence_max_degrees=weather_turbulence_max_degrees,
            initial_kick_delta_v=None if kick is None else kick[0],
            initial_kick_direction=None if kick is None else kick[1],
            lander_screen_position=_lander_screen_position(env),
        )


def _world_name(world) -> str | None:
    if world is None:
        return None
    name = getattr(world, "name", None)
    if name is None:
        return None
    return str(name)


def _world_gravity(world) -> float | None:
    if world is None:
        return None
    if _world_name(world) is None:
        return None
    gravity = getattr(world, "gravity", None)
    if gravity is None:
        return None
    try:
        return float(gravity)
    except TypeError:
        return None


def _weather_turbulence_max(env, inertia) -> float | None:
    weather = getattr(env, "_weather", None)
    if weather is None:
        return None
    _wind, turbulence = weather
    if not inertia:
        return None
    return math.degrees(abs(float(turbulence)) / float(inertia))


def _turbulence_acceleration(env) -> tuple[float, float] | None:
    env = getattr(env, "unwrapped", env)
    lander = getattr(env, "lander", None)
    inertia = getattr(lander, "inertia", None)
    torque_idx = getattr(env, "torque_idx", None)
    if not getattr(env, "enable_wind", False) or lander is None or not inertia or torque_idx is None:
        return None

    turbulence_power = float(getattr(env, "turbulence_power", 0.0))
    max_acceleration = _max_acceleration(turbulence_power, inertia)
    return _force_wave(torque_idx) * turbulence_power / float(inertia), max_acceleration


def _force_wave(index) -> float:
    return math.tanh(math.sin(0.02 * int(index)) + math.sin(math.pi * 0.01 * int(index)))


def _max_acceleration(force: float, divisor: float) -> float:
    if not divisor:
        return 0.0
    return abs(float(force)) / float(divisor)


def _lander_screen_position(env) -> tuple[int, int] | None:
    env = getattr(env, "unwrapped", env)
    lander = getattr(env, "lander", None)
    if lander is None:
        return None
    return (
        int(lander.position.x * lunar_lander.SCALE),
        int(lunar_lander.VIEWPORT_H - lander.position.y * lunar_lander.SCALE),
    )


def _initial_kick(seed: int | None, mass) -> tuple[float, tuple[float, float]] | None:
    if seed is None or not mass:
        return None
    fx, fy = _initial_force(seed)
    delta_v = float(math.hypot(fx, fy)) * (1.0 / lunar_lander.FPS) / float(mass)
    return delta_v, _kick_direction(fx, fy)


def _kick_direction(fx: float, fy: float) -> tuple[float, float]:
    length = math.hypot(fx, fy)
    if not length:
        return 0.0, 0.0
    return fx / length, fy / length


def _initial_force(seed: int) -> tuple[float, float]:
    rng, _ = seeding.np_random(seed)
    viewport_height = lunar_lander.VIEWPORT_H / lunar_lander.SCALE
    chunks = 11
    rng.uniform(0, viewport_height / 2, size=(chunks + 1,))
    fx = rng.uniform(-lunar_lander.INITIAL_RANDOM, lunar_lander.INITIAL_RANDOM)
    fy = rng.uniform(-lunar_lander.INITIAL_RANDOM, lunar_lander.INITIAL_RANDOM)
    return float(fx), float(fy)
