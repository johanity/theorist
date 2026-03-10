"""Persistent cross-domain theory store.

Every experiment updates the brain. Every domain confirms or refutes beliefs.
The brain compounds knowledge across sessions, tasks, and domains.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_THEORY = {
    "confirmed": [],
    "refuted": [],
    "open_questions": [],
    "meta_patterns": [],
    "cross_domain": [],
    "total_experiments": 0,
    "total_tasks": 0,
    "domains_seen": [],
    "avg_surprise": 0.0,
    "surprise_by_domain": {},
    "param_position_scores": {},
    "mid_vs_extreme_total": 0,
    "mid_vs_extreme_wins": 0,
    "prediction_errors": [],
}


class Brain:
    """Accumulated wisdom from every experiment ever run."""

    def __init__(self, path: str = "~/.theorist") -> None:
        self.path = Path(os.path.expanduser(path))
        self.path.mkdir(parents=True, exist_ok=True)
        self.theory_file = self.path / "theory.json"
        self.experiments_file = self.path / "experiments.jsonl"
        self.theory: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.theory_file.exists():
            return json.loads(self.theory_file.read_text())
        return {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                for k, v in _DEFAULT_THEORY.items()}

    def save(self) -> None:
        self.theory_file.write_text(json.dumps(self.theory, indent=2))

    def start_task(self, domain: str) -> str:
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if domain not in self.theory["domains_seen"]:
            self.theory["domains_seen"].append(domain)
            self.theory["total_tasks"] += 1
            self.theory["surprise_by_domain"][domain] = []
        self.save()
        return task_id

    def log_experiment(self, experiment: Dict[str, Any]) -> None:
        experiment["timestamp"] = datetime.now().isoformat()
        experiment["global_id"] = self.theory["total_experiments"]

        with open(self.experiments_file, "a") as f:
            f.write(json.dumps(experiment) + "\n")

        self.theory["total_experiments"] += 1

        if "surprise" in experiment:
            n = self.theory["total_experiments"]
            self.theory["avg_surprise"] = (
                self.theory["avg_surprise"] * (n - 1) + experiment["surprise"]
            ) / n
            domain = experiment.get("domain", "unknown")
            if domain in self.theory["surprise_by_domain"]:
                self.theory["surprise_by_domain"][domain].append(experiment["surprise"])

        self.save()

    def update_position_scores(self, position: str, metric: float) -> None:
        scores = self.theory["param_position_scores"]
        if position not in scores:
            scores[position] = {"total": 0.0, "count": 0, "avg_metric": 0.0}
        entry = scores[position]
        entry["total"] += metric
        entry["count"] += 1
        entry["avg_metric"] = entry["total"] / entry["count"]

    def record_mid_vs_extreme(self, is_mid: bool) -> None:
        self.theory["mid_vs_extreme_total"] += 1
        if is_mid:
            self.theory["mid_vs_extreme_wins"] += 1

    def record_prediction_error(self, task_index: int, exp_num: int,
                                error: float, surprise: float) -> None:
        self.theory["prediction_errors"].append({
            "task_index": task_index,
            "exp": exp_num,
            "error": error,
            "surprise": surprise,
        })

    def apply_update(self, update_type: str, content: str) -> None:
        dispatch = {
            "CONFIRM": lambda: self.theory["confirmed"].append(content)
                               if content not in self.theory["confirmed"] else None,
            "REFUTE": lambda: (
                self.theory["refuted"].append(content),
                self.theory["confirmed"].__init__(
                    [c for c in self.theory["confirmed"]
                     if content.lower() not in c.lower()]
                ),
            ),
            "ADD": lambda: self.theory["confirmed"].append(content)
                           if content not in self.theory["confirmed"] else None,
            "QUESTION": lambda: self.theory["open_questions"].append(content)
                                if content not in self.theory["open_questions"] else None,
            "META": lambda: self.theory["meta_patterns"].append(content)
                            if content not in self.theory["meta_patterns"] else None,
            "CROSS_DOMAIN": lambda: self.theory["cross_domain"].append(content)
                                    if content not in self.theory["cross_domain"] else None,
        }
        fn = dispatch.get(update_type)
        if fn:
            fn()
        self.save()

    def summary(self) -> str:
        t = self.theory
        lines = [
            "# Theorist Brain",
            f"**{t['total_experiments']} experiments across {t['total_tasks']} domains**",
            f"**Domains:** {', '.join(t['domains_seen']) or 'none yet'}",
            "",
        ]

        sections = [
            ("Universal Patterns", t["cross_domain"], None),
            ("Confirmed Beliefs", t["confirmed"], 15),
            ("Meta-Patterns", t["meta_patterns"], None),
            ("Open Questions", t["open_questions"], 5),
        ]
        for title, items, limit in sections:
            if items:
                lines.append(f"## {title}")
                for item in (items[-limit:] if limit else items):
                    lines.append(f"- {item}")
                lines.append("")

        mid_total = t.get("mid_vs_extreme_total", 0)
        mid_wins = t.get("mid_vs_extreme_wins", 0)
        if mid_total > 0:
            lines.append("## Stats")
            lines.append(f"- Moderate beats extremes: {mid_wins}/{mid_total} "
                         f"({100 * mid_wins / mid_total:.0f}%)")
            lines.append(f"- Avg surprise: {t['avg_surprise']:.3f}")

        return "\n".join(lines)

    # Alias for backward compat and discoverability
    get_summary = summary

    def get_cross_domain_prior(self, position: str) -> float:
        scores = self.theory["param_position_scores"]
        if position in scores and scores[position]["count"] > 5:
            return scores[position]["avg_metric"]
        return 0.0

    @property
    def mid_preference(self) -> float:
        total = self.theory.get("mid_vs_extreme_total", 0)
        wins = self.theory.get("mid_vs_extreme_wins", 0)
        if total < 5:
            return 0.5
        return wins / total

    def export_anonymized(self) -> Dict[str, Any]:
        return {
            "position_scores": self.theory["param_position_scores"],
            "meta_beliefs": self.theory["meta_patterns"],
            "cross_domain": self.theory["cross_domain"],
            "domains_seen": self.theory["domains_seen"],
            "total_experiments": self.theory["total_experiments"],
            "avg_surprise": self.theory["avg_surprise"],
            "mid_vs_extreme": {
                "total": self.theory.get("mid_vs_extreme_total", 0),
                "wins": self.theory.get("mid_vs_extreme_wins", 0),
            },
        }

    def reset(self) -> None:
        import shutil
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.theory = self._load()
