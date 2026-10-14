import numpy as np

from crowdtarget import neighbours as N


def test_iou_tracker_certain_matches_and_ambiguity():
    P = np.array([0, 0, 50, 100.0])
    Q = np.array([200, 0, 250, 100.0])
    tr = N.IouTracker()
    for t in range(1, 11):
        tr.step(t, np.stack([P, Q]))
    R = np.array([30, 0, 80, 100.0])            # overlaps P at IoU 0.25 from frame 11
    for t in range(11, 14):
        tr.step(t, np.stack([P, Q, R]))
    tracks = tr.tracks
    assert len(tracks) == 3
    tp = next(v for v in tracks.values() if np.allclose(v["boxes"][0], P))
    tq = next(v for v in tracks.values() if np.allclose(v["boxes"][0], Q))
    assert N.certain_frac(tp, 2, 10) == 1.0 and N.certain_frac(tq, 2, 13) == 1.0
    assert N.certain_frac(tp, 11, 13) == 0.0     # ambiguous while R overlaps
    assert N.certain_frac(tp, 1, 1) == 0.0       # the first frame of a track is never certain


def test_track_ends_on_a_miss():
    P = np.array([0, 0, 50, 100.0])
    tr = N.IouTracker()
    tr.step(1, P[None])
    tr.step(2, P[None])
    tr.step(3, np.zeros((0, 4)))
    tr.step(4, P[None])
    assert len(tr.tracks) == 2


def test_rule_violations_counts_runs_with_two_identities():
    tracks = {0: {"frames": list(range(20)), "boxes": [None] * 20, "certain": [True] * 20},
              1: {"frames": list(range(20)), "boxes": [None] * 20, "certain": [True] * 20}}
    ids = {0: [5] * 20, 1: [5] * 10 + [7] * 10}
    r = N.rule_violations(tracks, ids)
    assert r["runs"] == 2 and r["violated_runs"] == 1


def test_detector_recall_and_gt_ids():
    gt = np.array([[1, 3, 0, 0, 50, 100, 1, 1, 1.0], [1, 4, 200, 0, 50, 100, 1, 1, 1.0]])
    dets = {1: np.array([[2, 0, 52, 100.0]])}
    r = N.detector_recall(dets, gt, [1])
    assert r["gt_boxes"] == 2 and r["detected"] == 1
    ids = N.gt_id_per_detection(dets[1], gt[gt[:, 0] == 1])
    assert ids.tolist() == [3]
