import numpy as np

from crowdtarget import sliver as SL


def textured_scene(seed=0, shift=(0, 0)):
    rng = np.random.default_rng(seed)
    lum = 70 + rng.normal(0, 3, (200, 200))
    yy, xx = np.mgrid[:200, :200]
    person = ((yy - 100) / 45) ** 2 + ((xx - 100) / 18) ** 2 <= 1          # an upright ellipse
    pattern = 60 * np.sin(yy / 3.0) * np.cos(xx / 4.5) + 40 * ((yy // 9 + xx // 7) % 2)   # non-repeating-ish cloth
    lum = np.where(person, 110 + pattern, lum)
    if shift != (0, 0):
        lum = np.roll(np.roll(lum, shift[0], 0), shift[1], 1)
        person = np.roll(np.roll(person, shift[0], 0), shift[1], 1)
    return lum, person


def test_mine_finds_unique_templates_inside_the_mask_only():
    lum, mask = textured_scene()
    ts = SL.mine(lum, mask, max_n=8)
    assert 3 <= len(ts) <= 8
    assert all(mask[t["y"], t["x"]] for t in ts)
    assert all(max(abs(a["y"] - b["y"]), abs(a["x"] - b["x"])) >= 8 for i, a in enumerate(ts) for b in ts[i + 1:])


def test_step_certifies_on_the_shifted_copy_and_not_on_another_texture():
    lum, mask = textured_scene()
    ts = SL.mine(lum, mask, max_n=6)
    lum2, mask2 = textured_scene(shift=(4, -3))
    found = SL.step(lum2, ts, search_r=10)
    assert len(found) >= len(ts) // 2
    for f in found:
        assert (f["y"] - f["from"][0], f["x"] - f["from"][1]) == (4, -3)
    other = 70 + np.random.default_rng(9).normal(0, 25, (200, 200))        # noise, no structure in common
    assert SL.step(other, ts, search_r=10) == []


def test_chain_advances_and_counts_blackouts():
    lum, mask = textured_scene()
    ch = SL.Chain(max_n=6)
    assert ch.seed(lum, mask) > 0
    lum2, mask2 = textured_scene(shift=(3, 2))
    found = ch.advance(lum2, mask2)
    assert found and ch.length == 1 and ch.inside(found, mask2)
    blank = 70 + np.random.default_rng(3).normal(0, 3, (200, 200))
    assert ch.advance(blank) == [] and ch.broken_for == 1 and ch.length == 0
    lum3, mask3 = textured_scene(shift=(10, 6))                                  # back after one blank frame, 7 px further
    found = ch.advance(lum3, mask3)
    assert found and ch.broken_for == 0
