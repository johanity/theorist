"""The @theorist.experiment decorator -- simplest possible API."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict

from .core import Theorist
from .types import Results


def experiment(
    search_space: Dict[str, list],
    metric: str = "metric",
    minimize: bool = True,
    domain: str = "general",
    engine: str = "local",
) -> Callable:
    """Turn any function into an optimizable experiment.

    Usage:
        @experiment(
            search_space={"lr": [1e-4, 1e-3, 1e-2], "layers": [2, 4, 8]},
            metric="loss",
            minimize=True,
            domain="ml_training",
        )
        def train(config):
            return {"loss": 0.5}

        results = train.optimize(n=20)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(config: Dict[str, Any]) -> Any:
            return fn(config)

        def optimize(
            n: int = 20,
            verbose: bool = True,
            brain_path: str = "~/.theorist",
        ) -> Results:
            t = Theorist(
                domain=domain,
                brain_path=brain_path,
                engine=engine,
            )
            return t.optimize(
                fn=fn,
                search_space=search_space,
                n=n,
                metric=metric,
                minimize=minimize,
                verbose=verbose,
            )

        wrapper.optimize = optimize  # type: ignore[attr-defined]
        wrapper._search_space = search_space  # type: ignore[attr-defined]
        return wrapper

    return decorator
