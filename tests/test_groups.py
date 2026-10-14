import numpy as np

from crowdtarget import groups as G


def test_companion_detected_and_far_walker_ignored():
    rows = []
    for t in range(1, 41):
        rows.append([t, 1, 100 + 3 * t, 200, 40, 100, 1, 1, 1.0])       # target
        rows.append([t, 2, 150 + 3 * t, 205, 40, 100, 1, 1, 1.0])       # beside, same velocity
        rows.append([t, 3, 100 + 3 * t, 600, 40, 100, 1, 1, 1.0])       # far
        rows.append([t, 4, 150 - 3 * t, 200, 40, 100, 1, 1, 1.0])       # crosses, opposite velocity
    gt = np.array(rows)
    c = G.companions(gt, 1, 1, 40)
    assert [x["id"] for x in c] == [2]
