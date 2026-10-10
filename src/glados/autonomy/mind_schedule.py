"""Timing policies for the mind scheduler; no policy performs domain work."""

from collections.abc import Callable
from dataclasses import dataclass
import math
import random
from typing import Protocol


class Schedule(Protocol):
    adaptive: bool

    def delay(self) -> float | None: ...
    def reset(self) -> None: ...


@dataclass
class FixedInterval:
    seconds: float
    adaptive: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Mind interval must be positive and finite")

    def delay(self) -> float:
        return self.seconds

    def reset(self) -> None:
        pass


class OnDemand:
    adaptive = False

    def delay(self) -> None:
        return None

    def reset(self) -> None:
        pass


class RandomAdaptive:
    """Draw once per observation; changing motion adjusts the same pending deadline."""

    adaptive = True

    def __init__(
        self,
        bounds: Callable[[], tuple[float, float]],
        activity: Callable[[], float],
        draw: Callable[[], float] = random.random,
    ) -> None:
        self.bounds, self.activity, self.draw = bounds, activity, draw
        self.quantile = 0.5

    def reset(self) -> None:
        self.quantile = self.draw()

    def delay(self) -> float:
        minimum, maximum = self.bounds()
        if not (0 < minimum <= maximum and math.isfinite(maximum)):
            raise ValueError("Invalid adaptive interval range")
        weight = max(0.0, min(1.0, self.activity()))
        quiet = self.quantile**0.25
        active = 1 - (1 - self.quantile) ** 0.25
        return minimum + (maximum - minimum) * (quiet * (1 - weight) + active * weight)


class AdaptiveInterval:
    """Read a live delay, for example a longer interval when the room is empty."""

    adaptive = True

    def __init__(self, delay: Callable[[], float]) -> None:
        self._delay = delay

    def delay(self) -> float:
        value = self._delay()
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Adaptive delay must be positive and finite")
        return value

    def reset(self) -> None:
        pass
