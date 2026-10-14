import numpy as np

from crowdtarget import anchors as A


def scene(seed=0, disc=True):
    rng = np.random.default_rng(seed)
    lum = 60 + rng.normal(0, 4, (240, 240))
    if disc:
        yy, xx = np.mgrid[:240, :240]
        lum[(yy - 120) ** 2 + (xx - 110) ** 2 <= 5 ** 2] += 160
    return lum


def test_mine_finds_the_unique_blob_and_certifies_on_a_shifted_copy_only():
    lum = scene()
    box = (80, 70, 140, 170)
    found = A.mine(lum, box, None, stride=3)
    assert found, "the blob should give at least one S = {0} anchor"
    assert all(a["kappa"][r] >= 1 for a in found for r in range(3))
    shifted = np.roll(np.roll(lum, 4, 0), -3, 1)
    assert A.certified_count(shifted, (77, 74, 137, 174), found) >= 1
    assert A.certified_count(scene(seed=5, disc=False), box, found) == 0


def test_keep_distinctive_drops_anchors_the_crowd_also_shows():
    lum = scene()
    box = (80, 70, 140, 170)
    found = A.mine(lum, box, None, stride=3)
    kept, n = A.keep_distinctive([dict(a) for a in found], lum, [box])      # the "other" is the same person
    assert n == 1 and kept == []
    kept, n = A.keep_distinctive([dict(a) for a in found], lum, [(170, 70, 230, 170)])
    assert len(kept) == len(found)
