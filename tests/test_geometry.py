import numpy as np

from crowdtarget import geometry as G


def synthetic_gt(a=0.2, b=10.0):
    rows = []
    rng = np.random.default_rng(1)
    for id_, rho in ((1, 1.0), (2, 1.2), (3, 0.9)):
        for t in range(1, 60):
            y_foot = 200 + 8 * t + rng.normal(0, 2)
            h = rho * (a * y_foot + b) * (1 + rng.normal(0, 0.01))
            rows.append([t, id_, 100 * id_, y_foot - h, 40, h, 1, 1, 1.0])
    return np.array(rows)


def test_height_model_and_rho_are_depth_invariant():
    gt = synthetic_gt()
    ab, rel = G.height_model(gt)
    assert rel < 0.2          # population spread of heights, which kappa absorbs
    r1, r2, r3 = (G.person_rho(gt, i, ab) for i in (1, 2, 3))
    assert abs(r2 / r1 - 1.2) < 0.05 and abs(r3 / r1 - 0.9) < 0.05
    rows = gt[gt[:, 1] == 2]
    rho_t = G.rho(rows[:, 5], rows[:, 3] + rows[:, 5], ab)
    assert rho_t.std() / rho_t.mean() < 0.03
