"""Statistics helpers: same estimator as llama-bench (sample stddev)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    sq_sum = sum(v * v for v in vals)
    n = len(vals)
    return math.sqrt(max(0.0, sq_sum / (n - 1) - m * m * n / (n - 1)))


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


@dataclass
class Summary:
    n: int
    mean: float
    std: float

    @property
    def fmt(self) -> str:
        return f"{self.mean:.2f} ± {self.std:.2f}"


def summarize(values: Iterable[float | None]) -> Summary:
    vals = [v for v in values if v is not None]
    return Summary(n=len(vals), mean=mean(vals), std=stdev(vals))
