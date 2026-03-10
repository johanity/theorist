"""Structured data types for the predict-surprise-learn loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Prediction:
    config: Dict[str, Any]
    predicted_metric: float
    hypothesis: str
    confidence: str = "low"  # low | medium | high


@dataclass
class Experiment:
    exp_id: int
    config: Dict[str, Any]
    predicted: float
    actual: float
    surprise: float
    hypothesis: str
    learning: str
    status: str  # "best" | "discard"
    description: str = ""


@dataclass
class Results:
    best_config: Dict[str, Any]
    best_metric: float
    experiments: List[Experiment] = field(default_factory=list)
    theory_updates: List[str] = field(default_factory=list)
    domain: str = ""

    def report(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            "THEORIST -- Results",
            f"Domain: {self.domain}",
            f"Best metric: {self.best_metric:.4f}",
            f"Total experiments: {len(self.experiments)}",
            f"{'=' * 60}\n",
            f"{'Exp':>4} {'Metric':>10} {'Predicted':>10} {'Surprise':>8} {'Status':>8}  Hypothesis",
            "-" * 72,
        ]

        for e in self.experiments:
            lines.append(
                f"{e.exp_id:>4} {e.actual:>10.4f} {e.predicted:>10.4f} "
                f"{e.surprise:>8.3f} {e.status:>8}  {e.hypothesis[:35]}"
            )

        # Surprise trajectory -- quartile averages
        n = len(self.experiments)
        if n >= 4:
            q = n // 4
            lines.append("\nSurprise trajectory:")
            for i, label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
                start = i * q
                end = (i + 1) * q if i < 3 else n
                chunk = self.experiments[start:end]
                avg = sum(e.surprise for e in chunk) / len(chunk) if chunk else 0
                bar = "#" * int(avg * 40)
                lines.append(f"  {label}: {avg:.3f} {bar}")

        # Prediction calibration -- how accuracy improved over time
        if n >= 4:
            half = n // 2
            first_half = self.experiments[:half]
            second_half = self.experiments[half:]
            first_mae = sum(abs(e.predicted - e.actual) for e in first_half) / len(first_half)
            second_mae = sum(abs(e.predicted - e.actual) for e in second_half) / len(second_half)
            lines.append("\nPrediction calibration:")
            lines.append(f"  First half MAE:  {first_mae:.4f}")
            lines.append(f"  Second half MAE: {second_mae:.4f}")
            if first_mae > 0:
                improvement = (1 - second_mae / first_mae) * 100
                lines.append(f"  Improvement:     {improvement:.0f}%")

        if self.theory_updates:
            lines.append("\nLearnings:")
            for u in self.theory_updates[-10:]:
                lines.append(f"  - {u}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()
