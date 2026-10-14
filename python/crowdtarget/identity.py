"""Thresholds with guarantees, and the layered identity set (Theorems 4, 6, 7 of Part IV).

conformal_threshold   the ceil((n+1)(1-alpha))-th smallest calibration score: a fresh exchangeable
                      score exceeds it with probability <= alpha (split conformal).
clopper_pearson_upper upper confidence bound of a rate from x successes in n trials.
layered               applies exclusion layers in order to a list of candidates and returns the
                      survivors after each layer and where the true person fell, if it did.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def conformal_threshold(scores, alpha: float) -> float:
    s = np.sort(np.asarray([v for v in scores if np.isfinite(v)], np.float64))
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= n else float("inf")


def clopper_pearson_upper(x: int, n: int, conf: float = 0.95) -> float:
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(beta.ppf(conf, x + 1, n - x))


def layered(cands: list[dict], true_id: int, layers: list[tuple[str, callable]]) -> dict:
    """layers: [(name, keep(cand) -> bool)].  Returns survivors per layer and the layer that dropped the true person."""
    alive = list(cands)
    counts = {"start": len(alive)}
    true_alive = any(c["id"] == true_id for c in alive)
    dropped_at = None if true_alive else "start"
    for name, keep in layers:
        alive = [c for c in alive if keep(c)]
        counts[name] = len(alive)
        if dropped_at is None and not any(c["id"] == true_id for c in alive):
            dropped_at = name
    return {"counts": counts, "survivors": [c["id"] for c in alive], "true_dropped_at": dropped_at}
