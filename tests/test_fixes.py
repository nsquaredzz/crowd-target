import numpy as np

from crowdtarget import fixes


def test_box_iou_and_mask_box():
    assert fixes.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert fixes.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 10 * 5 / (200 - 50)
    assert fixes.box_iou([0, 0, 10, 10], None) == 0.0
    m = np.zeros((8, 8), bool)
    m[2:5, 3:7] = True
    assert fixes.mask_box(m).tolist() == [3, 2, 7, 5]
    assert fixes.mask_box(np.zeros((4, 4), bool)) is None


def test_kalman_follows_constant_velocity():
    kf = fixes.KalmanBox([100, 100, 140, 200])
    for t in range(1, 30):
        kf.predict()
        kf.update([100 + 3 * t, 100 + t, 140 + 3 * t, 200 + t])
    pred = kf.predict()
    truth = np.array([100 + 90, 100 + 30, 140 + 90, 200 + 30])
    assert np.abs(pred - truth).max() < 4
    assert fixes.box_iou(pred, truth) > 0.9


def test_samurai_choose_prefers_motion_consistent_candidate():
    class P:  # no model needed for choose()
        sam_mask_decoder = None
    s = fixes.Samurai(P(), 100, 100, [10, 10, 20, 40])
    s.pred_box = np.array([10, 10, 20, 40], float)
    aff = np.array([0.80, 0.85, 0.3])
    boxes = [np.array([10, 10, 20, 40.0]), np.array([60, 10, 70, 40.0]), None]
    # candidate 1 has the higher affinity but zero overlap with the prediction
    assert s.choose(aff, boxes, None, 5.0) == 0
    s.alpha_kf = 0.0
    assert s.choose(aff, boxes, None, 5.0) == 1


def test_dam_distractor_measure():
    class P:
        sam_mask_decoder = None
    d = fixes.DamAnchors(P(), 100, 100, [0, 0, 10, 10])
    chosen = np.zeros((10, 10), bool)
    chosen[0:5, 0:4] = True            # area 20
    alt = chosen.copy()
    alt[6:10, 6:10] = True             # 16 extra px outside the chosen mask
    d.last = {"chosen": 0, "masks": np.stack([chosen, alt, chosen])}
    assert abs(d.distractor() - 16 / 20) < 1e-9
    d.last = {"chosen": 0, "masks": np.stack([chosen, chosen, chosen])}
    assert d.distractor() == 0.0
