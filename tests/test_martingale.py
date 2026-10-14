"""Validity (Ville) and power of the conformal test martingales, on simulated streams."""
import numpy as np
import pytest

from crowdtarget.martingale import (MixtureMartingale, OnlinePValues, SimpleJumper, SwitchDetector,
                                    cusum_learned_null, first_crossing, run_offline)


def test_online_pvalues_uniform_under_exchangeability():
    rng = np.random.default_rng(0)
    ps = []
    for trial in range(200):
        s = rng.normal(size=60)
        pv = OnlinePValues(s[:20], seed=trial)
        ps += [pv(v) for v in s[20:]]
    ps = np.asarray(ps)
    # Kolmogorov-Smirnov distance to uniform on 8000 draws should be tiny
    grid = np.sort(ps)
    ks = np.max(np.abs(grid - (np.arange(1, grid.size + 1) / grid.size)))
    assert ks < 0.02, ks


@pytest.mark.parametrize("kind", ["mixture", "jumper"])
@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_false_alarm_rate_within_ville(kind, alpha):
    rng = np.random.default_rng(1)
    n_trials, T = 2000, 150
    hits = 0
    for trial in range(n_trials):
        s = rng.normal(size=T + 30)
        w = run_offline(s, n_cal=30, kind=kind, seed=trial)
        hits += first_crossing(w, alpha) is not None
    rate = hits / n_trials
    slack = 2 * np.sqrt(alpha * (1 - alpha) / n_trials)
    assert rate <= alpha + slack, (kind, alpha, rate)


@pytest.mark.parametrize("kind", ["mixture", "jumper"])
def test_power_on_a_shift(kind):
    rng = np.random.default_rng(2)
    delays = []
    for trial in range(100):
        s = np.concatenate([rng.normal(size=60), rng.normal(loc=2.5, size=120)])
        w = run_offline(s, n_cal=30, kind=kind, seed=trial)
        c = first_crossing(w, 0.05)
        assert c is not None, (kind, trial)
        delays.append(c - 30)          # steps after the shift (the shift is at tested index 30)
    assert np.median(delays) < 25, (kind, np.median(delays))


def test_detector_alarm_bookkeeping():
    det = SwitchDetector(np.zeros(20) + np.arange(20) * 0.01, kind="mixture", alpha=0.05, seed=3)
    for i in range(40):
        det.update(5.0 + i)            # every score far above the pool
    assert det.alarmed and det.alarm_at is not None and det.max_wealth >= 20


def test_jumper_mean_is_one_under_uniform():
    # the jumper's steps are bounded multiplicative factors, so its sample mean is testable;
    # power martingales have infinite variance (E[p^-1] diverges) and are tested through Ville only
    rng = np.random.default_rng(4)
    vals = []
    for _ in range(3000):
        m = SimpleJumper()
        for p in rng.uniform(size=25):
            m.update(p)
        vals.append(m.wealth)
    assert abs(np.mean(vals) - 1.0) < 0.1, np.mean(vals)


def test_power_betting_function_integrates_to_one():
    # for each eps, int_0^1 eps p^(eps-1) dp = 1; the substitution u = p^eps makes the quadrature exact
    for eps in (0.05, 0.3, 0.7, 0.95):
        u = (np.arange(200000) + 0.5) / 200000
        p = u ** (1.0 / eps)
        # E over p ~ U(0,1) of b(p) equals the mean of b(p) dp/du over u, and dp/du = p / (eps u)
        val = np.mean(eps * p ** (eps - 1) * p / (eps * u))
        assert abs(val - 1.0) < 1e-3, (eps, val)


def test_cusum_reference_runs():
    s = np.concatenate([np.zeros(30), np.ones(30) * 3])
    S = cusum_learned_null(s + 1e-3 * np.arange(60), n_cal=30)
    assert S[-1] > S[0] and S.shape == (30,)
