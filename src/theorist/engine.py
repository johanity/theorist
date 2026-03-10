"""Statistical prediction engine with cross-domain transfer.

Zero dependencies beyond stdlib. Handles explore/exploit, prediction,
recording, and value classification -- all driven by accumulated brain knowledge.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .brain import Brain
from .surprise import SurpriseNormalizer
from .types import Prediction


class Engine:
    """Predicts configs, records outcomes, learns from surprise."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._normalizer = SurpriseNormalizer()
        self._history: List[Dict[str, Any]] = []
        self._param_effects: Dict[str, Dict[str, List[float]]] = {}
        self._interaction_effects: Dict[str, List[float]] = {}  # "p1=v1|p2=v2" -> metrics
        self._param_surprise: Dict[str, List[float]] = {}  # per-param surprise tracking
        self._best_metric: Optional[float] = None
        self._best_config: Dict[str, Any] = {}
        self._search_space: Dict[str, list] = {}
        self._baseline: Dict[str, Any] = {}
        self._task_index: int = 0

    def start_task(self, search_space: Dict[str, list],
                   baseline: Dict[str, Any], domain: str) -> None:
        self._search_space = search_space
        self._baseline = baseline.copy()
        self._best_metric = None
        self._best_config = baseline.copy()
        self._history = []
        self._param_effects = {}
        self._interaction_effects = {}
        self._param_surprise = {}
        self._normalizer = SurpriseNormalizer()
        self._task_index = len(self.brain.theory.get("domains_seen", []))

    def classify_value(self, param: str, value: Any) -> str:
        space = self._search_space.get(param, [])
        if not space:
            return "mid"
        try:
            sorted_space = sorted(space)
            idx = sorted_space.index(value) if value in sorted_space else len(sorted_space) // 2
        except TypeError:
            return "mid"
        frac = idx / max(len(sorted_space) - 1, 1)
        if frac < 0.33:
            return "low"
        if frac > 0.66:
            return "high"
        return "mid"

    def predict(self, experiment_num: int) -> Prediction:
        if experiment_num == 0:
            return Prediction(
                config=self._baseline.copy(),
                predicted_metric=0.0,
                hypothesis="Baseline measurement",
                confidence="low",
            )

        config = self._best_config.copy()
        n_lifetime = self.brain.theory.get("total_experiments", 0)

        # Adaptive exploration: fewer explore rounds as brain matures
        explore_budget = max(3, 8 - self._task_index * 2)

        if experiment_num <= explore_budget:
            config, hypothesis = self._explore(config, n_lifetime)
        else:
            config, hypothesis = self._exploit(config)

        predicted = self._predict_metric(config)
        confidence = "low" if n_lifetime < 20 else ("medium" if n_lifetime < 50 else "high")

        return Prediction(
            config=config,
            predicted_metric=predicted,
            hypothesis=hypothesis,
            confidence=confidence,
        )

    def _predict_metric(self, config: Dict[str, Any]) -> float:
        if self._best_metric is None:
            return 0.0
        if not self._param_effects:
            return self._best_metric

        # Predict delta from best_metric based on per-param effects
        # Weight recent observations more via exponential decay
        deltas = []
        for param, val in config.items():
            key = str(val)
            if param in self._param_effects and key in self._param_effects[param]:
                scores = self._param_effects[param][key]
                # Exponential decay: recent observations weighted more
                decay = 0.7
                weights = [decay ** (len(scores) - 1 - i) for i in range(len(scores))]
                w_sum = sum(weights)
                weighted_avg = sum(s * w for s, w in zip(scores, weights)) / w_sum if w_sum else scores[-1]
                deltas.append(weighted_avg - self._best_metric)

        # Check pairwise interaction effects
        interaction_deltas = []
        params = list(config.keys())
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                ikey = f"{params[i]}={config[params[i]]}|{params[j]}={config[params[j]]}"
                if ikey in self._interaction_effects:
                    scores = self._interaction_effects[ikey]
                    decay = 0.7
                    weights = [decay ** (len(scores) - 1 - k) for k in range(len(scores))]
                    w_sum = sum(weights)
                    weighted_avg = sum(s * w for s, w in zip(scores, weights)) / w_sum if w_sum else scores[-1]
                    interaction_deltas.append(weighted_avg - self._best_metric)

        if deltas or interaction_deltas:
            # Combine: individual param deltas + interaction corrections
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            avg_interaction = sum(interaction_deltas) / len(interaction_deltas) if interaction_deltas else 0.0
            # Interactions are a correction on top of independent effects
            return self._best_metric + avg_delta + avg_interaction * 0.5

        return self._best_metric

    def _surprise_weighted_param(self) -> str:
        """Pick a param to explore, weighted by avg surprise (higher = more likely)."""
        params = list(self._search_space.keys())
        if not self._param_surprise:
            return random.choice(params)

        # Compute weights: base weight + avg surprise per param
        weights = []
        for p in params:
            if p in self._param_surprise and self._param_surprise[p]:
                avg_s = sum(self._param_surprise[p]) / len(self._param_surprise[p])
                weights.append(1.0 + avg_s * 3.0)  # surprise amplifies selection
            else:
                weights.append(2.0)  # untested params get high weight too

        # Weighted random selection
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for p, w in zip(params, weights):
            cumulative += w
            if r <= cumulative:
                return p
        return params[-1]

    def _explore(self, config: Dict[str, Any],
                 n_lifetime: int) -> Tuple[Dict[str, Any], str]:
        if n_lifetime < 20:
            # Early: surprise-weighted random sampling
            n_mut = random.randint(1, min(3, len(self._search_space)))
            params = []
            for _ in range(n_mut):
                p = self._surprise_weighted_param()
                if p not in params:
                    params.append(p)
            if not params:
                params = [random.choice(list(self._search_space.keys()))]
            for p in params:
                config[p] = random.choice(self._search_space[p])
            return config, f"Exploring {', '.join(params)}"

        # Later: bias toward moderate values if cross-domain evidence supports it
        param = self._surprise_weighted_param()
        values = sorted(self._search_space[param])
        n = len(values)

        if self.brain.mid_preference > 0.55 and n >= 3:
            mid_start = max(0, n // 4)
            mid_end = min(n, 3 * n // 4 + 1)
            config[param] = random.choice(values[mid_start:mid_end])
            return config, (f"Cross-domain prior: moderate {param} "
                            f"(mid wins {self.brain.mid_preference:.0%})")

        config[param] = random.choice(values)
        return config, f"Exploring {param} (high surprise)"

    def _exploit(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        recent = [h["config"] for h in self._history[-3:]]

        # Best observed value per param
        best_per_param: Dict[str, Tuple[str, float]] = {}
        for param in self._search_space:
            if param not in self._param_effects:
                continue
            best_val, best_score = None, float("inf")
            for val_str, scores in self._param_effects[param].items():
                avg = sum(scores) / len(scores)
                if avg < best_score:
                    best_score = avg
                    best_val = val_str
            if best_val is not None:
                best_per_param[param] = (best_val, best_score)

        # Try combining all best values
        if best_per_param:
            combined = config.copy()
            changes = []
            for param, (val_str, _) in best_per_param.items():
                for v in self._search_space[param]:
                    if str(v) == val_str:
                        combined[param] = v
                        if v != self._best_config.get(param):
                            changes.append(f"{param}={v}")
                        break
            if changes and combined not in recent:
                return combined, f"Exploit combo: {', '.join(changes)}"

        # Try untested values near best
        untested = [
            (p, v) for p in self._search_space for v in self._search_space[p]
            if v != self._best_config.get(p)
            and (p not in self._param_effects or str(v) not in self._param_effects[p])
        ]
        if untested:
            param, val = random.choice(untested)
            config[param] = val
            return config, f"Exploit: untried {param}={val}"

        # Second-best values for diversity
        for param in self._search_space:
            if param not in self._param_effects:
                continue
            for val_str, _ in sorted(
                self._param_effects[param].items(),
                key=lambda x: sum(x[1]) / len(x[1]),
            ):
                candidate = config.copy()
                for v in self._search_space[param]:
                    if str(v) == val_str:
                        candidate[param] = v
                        break
                if candidate not in recent and candidate != config:
                    return candidate, f"Exploit: {param}={candidate[param]} (runner-up)"

        # Fallback
        param = random.choice(list(self._search_space.keys()))
        config[param] = random.choice(self._search_space[param])
        return config, f"Exhausted exploit, exploring {param}"

    def record(self, config: Dict[str, Any], predicted: float,
               actual: float, minimize: bool = True) -> Tuple[float, str, str]:
        self._normalizer.update(actual)
        surprise = self._normalizer.surprise(predicted, actual)

        is_better = (
            self._best_metric is None
            or (minimize and actual < self._best_metric)
            or (not minimize and actual > self._best_metric)
        )
        if is_better:
            self._best_metric = actual
            self._best_config = config.copy()
            status = "best"
        else:
            status = "discard"

        # Track per-param effects
        changed = []
        for k in config:
            if config[k] != self._baseline.get(k):
                changed.append(k)
                self._param_effects.setdefault(k, {}).setdefault(str(config[k]), []).append(actual)

        # Track pairwise interactions when 2+ params changed together
        if len(changed) >= 2:
            for ci in range(len(changed)):
                for cj in range(ci + 1, len(changed)):
                    ikey = f"{changed[ci]}={config[changed[ci]]}|{changed[cj]}={config[changed[cj]]}"
                    self._interaction_effects.setdefault(ikey, []).append(actual)

        # Track per-param surprise for exploration weighting
        for k in changed:
            self._param_surprise.setdefault(k, []).append(surprise)

        # Cross-domain position learning
        for k in config:
            position = self.classify_value(k, config[k])
            self.brain.update_position_scores(position, actual)

        if status == "best" and changed:
            for k in changed:
                self.brain.record_mid_vs_extreme(self.classify_value(k, config[k]) == "mid")

        self.brain.record_prediction_error(
            self._task_index, len(self._history), abs(predicted - actual), surprise,
        )

        # Generate learning summary
        if changed:
            learning = (f"{changed} improved metric to {actual:.4f}" if status == "best"
                        else f"{changed} did not improve (got {actual:.4f})")
        else:
            learning = f"Baseline: {actual:.4f}"

        self._history.append({
            "config": config, "actual": actual,
            "surprise": surprise, "status": status, "changed": changed,
        })

        self.brain.save()
        return surprise, learning, status

    @property
    def best_metric(self) -> Optional[float]:
        return self._best_metric

    @property
    def best_config(self) -> Optional[Dict[str, Any]]:
        return self._best_config
