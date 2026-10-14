import numpy as np

from crowdtarget.data.mot import targets, track
from crowdtarget.episodes import bin_of, dev_split, mine, visibility_series


def gt_rows(id_, frames, vis, x=100, y=100, w=40, h=120, cls=1, conf=1):
    return [[f, id_, x + 0.5 * f, y, w, h, conf, cls, v] for f, v in zip(frames, vis)]


def make_gt(n=300):
    rows = []
    # id 1: visible 1..100, gap 101..130 (vis 0.1), visible 131..300  -> one episode, g = 30
    rows += gt_rows(1, range(1, 101), [1.0] * 100)
    rows += gt_rows(1, range(101, 131), [0.1] * 30)
    rows += gt_rows(1, range(131, 301), [0.9] * 170)
    # id 2: gap of 10 only -> no episode
    rows += gt_rows(2, range(1, 151), [1.0] * 150)
    rows += gt_rows(2, range(161, 301), [1.0] * 140)
    # id 3: absent (no rows) 51..140, g = 90, visible 1..50 and 141..300 -> one episode in 80+
    rows += gt_rows(3, range(1, 51), [1.0] * 50)
    rows += gt_rows(3, range(141, 301), [1.0] * 160)
    # id 4: gap 101..125 but short post run (ends at 140) -> no episode
    rows += gt_rows(4, range(1, 101), [1.0] * 100)
    rows += gt_rows(4, range(101, 126), [0.0] * 25)
    rows += gt_rows(4, range(126, 141), [1.0] * 15)
    # id 5: too small (h = 30) -> no episode
    rows += gt_rows(5, range(1, 101), [1.0] * 100, h=30)
    rows += gt_rows(5, range(101, 131), [0.0] * 30, h=30)
    rows += gt_rows(5, range(131, 301), [1.0] * 170, h=30)
    # id 6: class 7 (static person) -> ignored
    rows += gt_rows(6, range(1, 301), [1.0] * 300, cls=7)
    return np.asarray(rows, np.float64)


def test_visibility_series_and_targets():
    gt = make_gt()
    v = visibility_series(gt, 3, 300)
    assert v[50] == 1.0 and v[51] == 0.0 and v[141] == 1.0
    assert 6 not in np.unique(targets(gt)[:, 1])
    assert track(gt, 1)[0, 0] == 1


def test_miner_finds_exactly_the_expected_episodes():
    gt = make_gt()
    eps = mine(gt, "synthetic", 300, 640, 480, 25.0)
    got = {(e.id, e.gap_start, e.gap_end, e.g, e.bin) for e in eps}
    assert got == {(1, 101, 130, 30, "20-40"), (3, 51, 140, 90, "80+")}, got
    e1 = [e for e in eps if e.id == 1][0]
    assert e1.t_start == 71 and e1.t_end == 190 and e1.n_frames == 120
    assert e1.start_box[3] - e1.start_box[1] == 120


def test_bins_and_split_are_deterministic_and_disjoint():
    assert bin_of(20) == "20-40" and bin_of(39) == "20-40" and bin_of(40) == "40-80" and bin_of(80) == "80+"
    gt = make_gt()
    eps = mine(gt, "synthetic", 300, 640, 480, 25.0)
    dev, held = dev_split(eps, per_bin=1)
    assert len(dev) == 2 and len(held) == 0
    dev2, _ = dev_split(eps, per_bin=1)
    assert [(e.id, e.gap_start) for e in dev] == [(e.id, e.gap_start) for e in dev2]
