import numpy as np

from crowdtarget import hindsight as HS


def test_first_long_loss():
    assert HS.first_long_loss([True] * 10) is None
    assert HS.first_long_loss([True, True, False, True] + [False] * 5 + [True]) == 3
    assert HS.first_long_loss([True, False, False, False, False, False]) == 0


def test_meet_scores_and_decide():
    F = np.zeros((3, 20, 20), bool)
    F[:, 5:10, 5:10] = True
    B = np.zeros((3, 3, 20, 20), bool)
    B[0, :, 5:10, 5:10] = True          # meets the target exactly
    B[1, :, 12:18, 12:18] = True        # elsewhere
    B[2, :, 6:11, 6:11] = True          # overlaps partly (IoU 16/34)
    m = HS.meet_scores(F, B)
    assert m[0] == 1.0 and m[1] == 0.0 and abs(m[2] - 16 / 34) < 1e-9
    d = HS.decide(m)
    assert d["kind"] == "ambiguous" and d["set"] == [0, 2]          # second-best 0.47 >= 0.25: no commit
    d = HS.decide(m[:2])
    assert d["kind"] == "commit" and d["index"] == 0
    assert HS.decide(np.array([0.1, 0.05]))["kind"] == "absent"
    assert HS.decide(np.array([]))["kind"] == "absent"
