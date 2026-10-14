import numpy as np

from crowdtarget.metrics import containment, episode_outcome, frame_outcome, summarise


def test_containment():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert containment(m, [5, 5, 15, 15]) == 1.0
    assert containment(m, [10, 5, 20, 15]) == 0.5
    assert containment(m, [0, 0, 2, 2]) == 0.0


def test_frame_outcome():
    gt = np.array([[1, 7, 0, 0, 10, 10, 1, 1, 1.0], [1, 8, 10, 0, 10, 10, 1, 1, 1.0]], np.float64)
    m = np.zeros((10, 20), bool)
    m[:, 12:18] = True
    assert frame_outcome(m, gt, 1, 7) == "switched:8"
    assert frame_outcome(m, gt, 1, 8) == "correct"
    assert frame_outcome(np.zeros((10, 20), bool), gt, 1, 7) == "absent"
    # a mask straddling both boxes and hanging below them: a quarter in each, so nowhere
    m2 = np.zeros((20, 20), bool)
    m2[:, 8:12] = True
    assert frame_outcome(m2, gt, 1, 7) == "wrong"


def test_episode_outcome_cases():
    frames = list(range(1, 41))
    vis = {t: True for t in range(21, 41)}
    # recovered: absent in the gap (11..20), correct after
    pf = ["correct"] * 10 + ["absent"] * 10 + ["correct"] * 20
    o = episode_outcome(pf, frames, 11, 20, vis)
    assert o["outcome"] == "recovered" and o["delay"] == 1
    # switched in the gap onto id 9 for 10 frames
    pf = ["correct"] * 10 + ["switched:9"] * 10 + ["correct"] * 20
    o = episode_outcome(pf, frames, 11, 20, vis)
    assert o["outcome"] == "switched" and o["switch_id"] == 9 and o["switch_phase"] == "gap" and o["switch_frame"] == 11
    # a 9-frame wrong run is not a switch; then lost because correct on < 50 % of post
    pf = ["correct"] * 10 + ["absent"] * 10 + ["switched:9"] * 9 + ["wrong"] * 11
    o = episode_outcome(pf, frames, 11, 20, vis)
    assert o["outcome"] == "lost"
    s = summarise([{"bin": "20-40", **o}, {"bin": "20-40", "outcome": "recovered", "switch_phase": None, "delay": 3}])
    assert s["all"]["n"] == 2 and s["20-40"]["lost"] == 1 and s["all"]["median_delay"] == 3.0
