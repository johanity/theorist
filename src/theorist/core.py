"""The Theorist -- main orchestrator for prediction-error optimization.

predict -> run -> surprise -> learn -> transfer
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .brain import Brain
from .engine import Engine
from .types import Experiment, Results


class Theorist:
    """Cross-domain prediction-error research agent."""

    def __init__(
        self,
        domain: str = "general",
        brain_path: str = "~/.theorist",
        engine: str = "local",
    ) -> None:
        self.domain = domain
        self.brain = Brain(path=brain_path)

        if engine == "smart":
            from .smart import SmartEngine
            self._engine: Engine = SmartEngine(self.brain)
        else:
            self._engine = Engine(self.brain)

    def optimize(
        self,
        fn: Callable[[Dict[str, Any]], Any],
        search_space: Dict[str, list],
        n: int = 20,
        metric: str = "metric",
        minimize: bool = True,
        baseline: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Results:
        if baseline is None:
            baseline = {k: v[len(v) // 2] for k, v in search_space.items()}

        self.brain.start_task(self.domain)
        self._engine.start_task(search_space, baseline, self.domain)

        experiments = []
        theory_updates = []

        if verbose:
            lifetime = self.brain.theory["total_experiments"]
            domains = self.brain.theory["total_tasks"]
            print(f"\n{'=' * 60}")
            print(f"THEORIST")
            print(f"Domain: {self.domain} | Lifetime: {lifetime} exp across {domains} domains")
            print(f"{'=' * 60}\n")

        for i in range(n):
            prediction = self._engine.predict(i)
            config = self._coerce_config(prediction.config, search_space, baseline)

            result = fn(config)
            actual = result[metric] if isinstance(result, dict) else float(result)

            surprise, learning, status = self._engine.record(
                config, prediction.predicted_metric, actual, minimize
            )

            changes = [f"{k}={config[k]}" for k in config if config.get(k) != baseline.get(k)]
            description = ", ".join(changes) if changes else "baseline"

            exp = Experiment(
                exp_id=i, config=config,
                predicted=prediction.predicted_metric, actual=actual,
                surprise=surprise, hypothesis=prediction.hypothesis,
                learning=learning, status=status, description=description,
            )
            experiments.append(exp)
            if learning:
                theory_updates.append(learning)

            self.brain.log_experiment({
                "domain": self.domain, "exp_id": i, "config": config,
                "predicted": prediction.predicted_metric, "actual": actual,
                "surprise": surprise, "status": status, "description": description,
            })

            if verbose:
                marker = " ***" if status == "best" else ""
                print(f"  Exp {i:2d}: {actual:.4f} "
                      f"(pred={prediction.predicted_metric:.4f}, "
                      f"surp={surprise:.3f}) "
                      f"{description[:40]}{marker}")

        results = Results(
            best_config=self._engine.best_config or baseline,
            best_metric=self._engine.best_metric or 0.0,
            experiments=experiments,
            theory_updates=theory_updates,
            domain=self.domain,
        )

        if verbose:
            print(f"\nBest: {results.best_metric:.4f}")
            print(f"Lifetime experiments: {self.brain.theory['total_experiments']}")

        return results

    @staticmethod
    def _coerce_config(config: Dict[str, Any], search_space: Dict[str, list],
                       baseline: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in config.items():
            if k in search_space and search_space[k]:
                expected = type(search_space[k][0])
                if not isinstance(v, expected):
                    try:
                        config[k] = expected(v)
                    except (ValueError, TypeError):
                        config[k] = baseline[k]
        return config

    @property
    def theory(self) -> str:
        return self.brain.summary()
