"""
Prediction-error optimization. Single-file, zero dependencies.
Usage: from theorist import optimize

Author: Johan David Bonilla (https://github.com/johanity)
Based on: https://github.com/johanity/epistemic-autoresearch
"""

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these directly
# ---------------------------------------------------------------------------

BRAIN_PATH = os.path.expanduser("~/.theorist")  # where the brain persists

@dataclass
class Config:
    n: int = 20                  # experiments to run
    metric: str = "metric"       # key in fn return dict
    minimize: bool = True        # True = lower is better
    domain: str = "default"      # for cross-domain transfer
    explore_frac: float = 0.4    # fraction of experiments spent exploring

# ---------------------------------------------------------------------------
# Surprise — Welford's online algorithm, scale-invariant
# ---------------------------------------------------------------------------

class Surprise:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n >= 2 else 1.0

    def measure(self, predicted, actual):
        err = abs(predicted - actual)
        scale = max(self.std, abs(self.mean) * 0.01, 0.001)
        return min(err / scale, 1.0)

# ---------------------------------------------------------------------------
# Brain — persists across sessions
# ---------------------------------------------------------------------------

class Brain:
    def __init__(self, path=BRAIN_PATH):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.f = self.path / "theory.json"
        self.t = json.loads(self.f.read_text()) if self.f.exists() else {
            "n_exp": 0, "n_tasks": 0, "domains": [],
            "effects": {}, "mid_wins": 0, "mid_total": 0,
        }

    def save(self):
        self.f.write_text(json.dumps(self.t, indent=2))

    def start(self, domain):
        if domain not in self.t["domains"]:
            self.t["domains"].append(domain)
            self.t["n_tasks"] += 1
            self.save()

    def log(self, data):
        with open(self.path / "log.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
        self.t["n_exp"] += 1

    def effect(self, param, val, metric):
        e = self.t.setdefault("effects", {})
        e.setdefault(param, {}).setdefault(str(val), []).append(metric)

    def position(self, is_mid):
        self.t["mid_total"] += 1
        if is_mid:
            self.t["mid_wins"] += 1

    @property
    def mid_pref(self):
        t = self.t["mid_total"]
        return self.t["mid_wins"] / t if t >= 5 else 0.5

    def summary(self):
        t = self.t
        s = f"brain: {t['n_exp']} exp, {t['n_tasks']} domains"
        if t["mid_total"] > 0:
            s += f", moderate beats extremes {100*t['mid_wins']//t['mid_total']}%"
        return s

    def reset(self):
        import shutil
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.__init__(str(self.path))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(val, space):
    """low/mid/high position in search space."""
    try:
        s = sorted(space)
        idx = s.index(val) if val in s else len(s) // 2
    except TypeError:
        return "mid"
    frac = idx / max(len(s) - 1, 1)
    return "low" if frac < 0.33 else "high" if frac > 0.66 else "mid"


def _pred(best, effects, config):
    """Estimate metric from per-param history. Exponential recency weighting."""
    if best is None or not effects:
        return best or 0.0
    deltas = []
    for p, v in config.items():
        k = str(v)
        if p in effects and k in effects[p]:
            scores = effects[p][k]
            w = [0.7 ** (len(scores)-1-i) for i in range(len(scores))]
            ws = sum(w)
            deltas.append(sum(s*wi for s, wi in zip(scores, w)) / ws - best)
    return best + (sum(deltas) / len(deltas) if deltas else 0.0)

# ---------------------------------------------------------------------------
# optimize() — the whole thing
# ---------------------------------------------------------------------------

def optimize(fn, space, n=20, metric="metric", minimize=True,
             domain="default", brain_path=BRAIN_PATH, verbose=True):
    """
    Run n experiments with prediction-error learning.

    fn: takes config dict, returns dict with metric key
    space: {"param": [val1, val2, ...]}
    """
    brain = Brain(brain_path)
    brain.start(domain)
    surp = Surprise()

    # state
    effects = {}
    best_m = None
    best_c = {k: v[len(v)//2] for k, v in space.items()}
    exps = []
    n_explore = int(n * Config.explore_frac)

    if verbose:
        print(f"\ntheor  {domain}  n={n}  {brain.summary()}")

    for i in range(n):

        # ---- predict ----
        if i == 0:
            c, pred, hyp = best_c.copy(), 0.0, "baseline"
        elif i <= n_explore:
            c = best_c.copy()
            p = random.choice(list(space.keys()))
            c[p] = random.choice(space[p])
            pred = _pred(best_m, effects, c)
            hyp = f"explore {p}={c[p]}"
        else:
            c = best_c.copy()
            for p in space:
                if p in effects:
                    bv, bs = None, float("inf")
                    for vs, sc in effects[p].items():
                        a = sum(sc)/len(sc)
                        if a < bs:
                            bs, bv = a, vs
                    if bv:
                        for v in space[p]:
                            if str(v) == bv:
                                c[p] = v
                                break
            pred = _pred(best_m, effects, c)
            hyp = "exploit"

        # ---- run ----
        result = fn(c)
        actual = result[metric] if isinstance(result, dict) else float(result)

        # ---- surprise ----
        surp.update(actual)
        s = surp.measure(pred, actual) if i > 0 else 0.0

        # ---- learn ----
        better = (best_m is None
                  or (minimize and actual < best_m)
                  or (not minimize and actual > best_m))
        if better:
            best_m, best_c = actual, c.copy()
            status = "best"
        else:
            status = "discard"
        if i == 0:
            status = "baseline"

        for k, v in c.items():
            effects.setdefault(k, {}).setdefault(str(v), []).append(actual)
            brain.effect(k, str(v), actual)
            if better:
                brain.position(_pos(v, space[k]) == "mid")

        brain.log({"domain": domain, "i": i, "actual": actual,
                   "pred": pred, "surprise": s, "status": status})
        exps.append({"i": i, "actual": actual, "pred": pred, "s": s, "status": status})

        if verbose:
            mark = " ***" if status == "best" else ""
            print(f"  {i:3d}  {actual:8.4f}  pred={pred:8.4f}  "
                  f"surp={s:.3f}  {hyp[:28]}{mark}")

    # ---- report ----
    brain.save()
    if verbose and len(exps) >= 4:
        h = len(exps)//2
        mae1 = sum(abs(e["pred"]-e["actual"]) for e in exps[:h]) / h
        mae2 = sum(abs(e["pred"]-e["actual"]) for e in exps[h:]) / (len(exps)-h)
        print(f"\n  prediction: MAE {mae1:.4f} → {mae2:.4f}", end="")
        if mae1 > 0:
            print(f" ({(1-mae2/mae1)*100:+.0f}%)", end="")
        print(f"  best: {best_m:.4f}")

    return {"best_config": best_c, "best_metric": best_m, "experiments": exps}


def compare(fn, space, n=20, metric="metric", minimize=True, domain="compare"):
    """Side by side: theorist vs random search."""
    print("---- theorist ----")
    t = optimize(fn, space, n, metric, minimize, domain, "/tmp/.theorist-cmp")
    print("\n---- random ----")
    rb = None
    for i in range(n):
        c = {k: random.choice(v) for k, v in space.items()}
        r = fn(c)
        a = r[metric] if isinstance(r, dict) else float(r)
        better = rb is None or (minimize and a < rb) or (not minimize and a > rb)
        if better:
            rb = a
        mark = " ***" if better else ""
        print(f"  {i:3d}  {a:8.4f}{mark}")
    print(f"\n  theorist: {t['best_metric']:.4f}  random: {rb:.4f}")
    return t, rb
