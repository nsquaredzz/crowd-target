import numpy as np

from crowdtarget import identity as I


def test_conformal_threshold_covers():
    rng = np.random.default_rng(0)
    alpha = 0.1
    exceed = []
    for _ in range(300):
        cal = rng.standard_normal(40)
        q = I.conformal_threshold(cal, alpha)
        exceed.append(rng.standard_normal(50) > q)
    rate = np.mean(np.concatenate(exceed))
    assert rate <= alpha + 0.02
    assert I.conformal_threshold([1, 2, 3], 0.01) == float("inf")      # too few points for that level


def test_clopper_pearson_upper():
    assert abs(I.clopper_pearson_upper(0, 20) - (1 - 0.05 ** (1 / 20))) < 1e-9
    assert I.clopper_pearson_upper(5, 5) == 1.0
    assert 0.1 < I.clopper_pearson_upper(1, 20) < 0.3


def test_layered_reports_where_the_true_person_fell():
    cands = [{"id": 1, "a": 0.1, "b": 0.5}, {"id": 2, "a": 0.9, "b": 0.5}, {"id": 3, "a": 0.1, "b": 0.9}]
    r = I.layered(cands, 1, [("a", lambda c: c["a"] < 0.5), ("b", lambda c: c["b"] < 0.8)])
    assert r["counts"] == {"start": 3, "a": 2, "b": 1} and r["survivors"] == [1] and r["true_dropped_at"] is None
    r = I.layered(cands, 2, [("a", lambda c: c["a"] < 0.5)])
    assert r["true_dropped_at"] == "a"
