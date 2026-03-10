"""LLM-powered hypothesis generation via Anthropic API.

Falls back gracefully to the local engine when the API is unavailable.
Requires: pip install theorist[smart]
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .brain import Brain
from .engine import Engine
from .types import Prediction


class SmartEngine(Engine):
    """Engine enhanced with Claude-powered reasoning."""

    def __init__(self, brain: Brain, model: str = "claude-haiku-4-5-20251001") -> None:
        super().__init__(brain)
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
            self._client = anthropic.Anthropic()
            return self._client
        except (ImportError, Exception):
            return None

    def predict(self, experiment_num: int) -> Prediction:
        base = super().predict(experiment_num)
        if experiment_num < 2:
            return base

        client = self._get_client()
        if client is None:
            return base

        history_str = "\n".join(
            f"  config={h['config']} -> metric={h['actual']:.4f} ({h['status']})"
            for h in self._history[-10:]
        )

        prompt = (
            "You are a research agent optimizing an experiment.\n\n"
            f"Search space: {json.dumps(self._search_space)}\n"
            f"Baseline: {json.dumps(self._baseline)}\n"
            f"Best so far: {self._best_metric:.4f} with config {json.dumps(self._best_config)}\n"
            f"Experiment #{experiment_num}\n\n"
            f"Recent history:\n{history_str}\n\n"
            f"Brain knowledge:\n{self.brain.summary()}\n\n"
            "Suggest a config to try next.\n"
            'Respond with JSON only: {"config": {...}, "hypothesis": "...", "predicted_metric": N}'
        )

        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)

            config = data.get("config", base.config)
            for k, v in config.items():
                if k in self._search_space and v not in self._search_space[k]:
                    try:
                        config[k] = min(self._search_space[k], key=lambda x: abs(x - v))
                    except (TypeError, ValueError):
                        config[k] = self._baseline.get(k, v)

            return Prediction(
                config=config,
                predicted_metric=data.get("predicted_metric", base.predicted_metric),
                hypothesis=data.get("hypothesis", base.hypothesis),
                confidence="high",
            )
        except Exception:
            return base
