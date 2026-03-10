"""Scale-invariant surprise via Welford's online algorithm.

Makes prediction error comparable across domains where metrics live on
wildly different scales (loss=0.03 vs latency=450ms vs reward=-12.7).
"""

from __future__ import annotations

import math


class SurpriseNormalizer:
    """Running statistics that turn raw prediction error into [0, 1] surprise."""

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return math.sqrt(self._m2 / (self._n - 1))

    @property
    def count(self) -> int:
        return self._n

    def surprise(self, predicted: float, actual: float) -> float:
        error = abs(predicted - actual)
        scale = max(self.std, abs(self._mean) * 0.01, 0.001)
        return min(error / scale, 1.0)
