"""Conformal test martingales for detecting that a tracked person is no longer the same person.

The object under test is a stream of nonconformity scores s_1, s_2, ... (large = unlike the
target's own history).  Under the null hypothesis that the stream is exchangeable with the
calibration pool, the online conformal p-values

    p_t = ( #{i : s_i > s_t} + U_t * (#{i : s_i = s_t} + 1) ) / (n_t + 1),   U_t ~ Uniform(0, 1),

with the pool grown by s_t after each step, are independent and uniform (Vovk, Gammerman and
Shafer, "Algorithmic Learning in a Random World", 2005).  A test martingale is
W_t = prod_{u <= t} b(p_u) for a betting function b >= 0 with integral one over [0, 1]; then
E[W_t] = 1 under the null and Ville's inequality gives

    P( sup_t W_t >= 1 / alpha ) <= alpha.

So the alarm "W_t >= 1/alpha" is a time-uniform test of exchangeability at level alpha, with
no parametric model and no training.  Two betting functions from Vovk et al., "Retrain or not
retrain: conformal test martingales for change-point detection" (2021):

  mixture       W_t = int_0^1 prod_u eps p_u^(eps - 1) d eps   (the simple mixture of power
                martingales; computed on a grid of eps in log space)
  simple jumper capitals C_e, e in {-1, 0, 1}, each a third at the start; at every step
                C_e <- (1 - J) C_e + (J / 3) sum C, then C_e <- C_e (1 + e (p - 1/2)); W = sum C

The learned-null CUSUM at the end is the comparison, not a claim: it standardises scores by the
calibration mean and deviation and accumulates max(0, S + z - k).

Scores on consecutive frames are correlated, so exchangeability is at best approximate on a
subsampled stream; the empirical false-alarm rate is measured and reported beside alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- p-values


class OnlinePValues:
    """Conformal p-values against a pool that grows with every tested score."""

    def __init__(self, calibration: np.ndarray, seed: int = 0, grow: bool = True):
        self.pool = list(np.asarray(calibration, np.float64).ravel())
        self.rng = np.random.default_rng(seed)
        self.grow = grow

    def __call__(self, s: float) -> float:
        pool = np.asarray(self.pool)
        n = pool.size
        gt = int(np.sum(pool > s))
        eq = int(np.sum(pool == s))
        p = (gt + self.rng.uniform() * (eq + 1)) / (n + 1)
        if self.grow:
            self.pool.append(float(s))
        return float(p)


# ----------------------------------------------------------------------------- martingales


class MixtureMartingale:
    """Simple mixture of power martingales, log-space on a grid of eps."""

    def __init__(self, n_grid: int = 200):
        self.eps = (np.arange(n_grid) + 0.5) / n_grid        # midpoints of [0, 1]
        self.logw = np.zeros(n_grid)                          # log prod eps p^(eps-1), per eps
        self.t = 0

    def update(self, p: float) -> float:
        p = min(max(float(p), 1e-12), 1.0)
        self.logw += np.log(self.eps) + (self.eps - 1.0) * np.log(p)
        self.t += 1
        return self.wealth

    @property
    def wealth(self) -> float:
        m = self.logw.max()
        return float(np.exp(m) * np.mean(np.exp(self.logw - m)))

    @property
    def log_wealth(self) -> float:
        m = self.logw.max()
        return float(m + np.log(np.mean(np.exp(self.logw - m))))


class SimpleJumper:
    """Vovk's simple jumper: three betting states with capital that jumps between them."""

    def __init__(self, jump: float = 0.01):
        self.J = float(jump)
        self.C = np.full(3, 1.0 / 3.0)
        self.e = np.array([-1.0, 0.0, 1.0])
        self.t = 0

    def update(self, p: float) -> float:
        tot = self.C.sum()
        self.C = (1 - self.J) * self.C + (self.J / 3.0) * tot
        self.C = self.C * (1.0 + self.e * (float(p) - 0.5))
        self.t += 1
        return self.wealth

    @property
    def wealth(self) -> float:
        return float(self.C.sum())

    @property
    def log_wealth(self) -> float:
        return float(np.log(max(self.wealth, 1e-300)))


# ----------------------------------------------------------------------------- the detector


@dataclass
class SwitchDetector:
    """Online: feed nonconformity scores, get the wealth; alarm when wealth >= 1/alpha.

    kind: "mixture" or "jumper".  The pool starts from the calibration scores (the target's own
    confidently visible frames) and grows with every tested score, which is what makes the
    p-values iid uniform under exchangeability.
    """
    calibration: np.ndarray
    kind: str = "mixture"
    alpha: float = 0.05
    seed: int = 0
    jump: float = 0.01
    grow: bool = True
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.pv = OnlinePValues(self.calibration, self.seed, self.grow)
        self.mart = MixtureMartingale() if self.kind == "mixture" else SimpleJumper(self.jump)
        self.max_wealth = 1.0
        self.alarm_at: int | None = None

    def update(self, s: float) -> tuple[float, float]:
        p = self.pv(s)
        w = self.mart.update(p)
        self.history.append((float(s), p, w))
        if w > self.max_wealth:
            self.max_wealth = w
        if self.alarm_at is None and w >= 1.0 / self.alpha:
            self.alarm_at = len(self.history) - 1
        return p, w

    @property
    def alarmed(self) -> bool:
        return self.alarm_at is not None


def run_offline(scores: np.ndarray, n_cal: int, kind: str = "mixture", seed: int = 0, jump: float = 0.01,
                grow: bool = True) -> np.ndarray:
    """Wealth after each tested score, scores[n_cal:] tested against the first n_cal as the pool."""
    det = SwitchDetector(np.asarray(scores[:n_cal]), kind=kind, alpha=1.0, seed=seed, jump=jump, grow=grow)
    return np.array([det.update(s)[1] for s in np.asarray(scores[n_cal:], np.float64)])


def first_crossing(wealth: np.ndarray, alpha: float) -> int | None:
    """Index of the first step whose wealth reaches 1/alpha, or None."""
    hit = np.nonzero(np.asarray(wealth) >= 1.0 / alpha)[0]
    return int(hit[0]) if hit.size else None


# ----------------------------------------------------------------------------- the comparison


def cusum_learned_null(scores: np.ndarray, n_cal: int, k: float = 0.5) -> np.ndarray:
    """One-sided CUSUM on scores standardised by the calibration mean and deviation (the parametric
    comparison of certified-skip §6): S_t = max(0, S_{t-1} + z_t - k)."""
    s = np.asarray(scores, np.float64)
    cal = s[:n_cal]
    mu, sd = float(cal.mean()), float(max(cal.std(ddof=1) if cal.size > 1 else 1.0, 1e-6))
    out = np.zeros(s.size - n_cal)
    S = 0.0
    for i, v in enumerate(s[n_cal:]):
        S = max(0.0, S + (v - mu) / sd - k)
        out[i] = S
    return out
