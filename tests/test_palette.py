import numpy as np

from crowdtarget import palette as P

RED, BLUE, GREEN = np.array([200, 50, 50.0]), np.array([50, 50, 200.0]), np.array([50, 200, 50.0])


def person(n_red, n_blue, n_green=0):
    return np.concatenate([np.tile(RED, (n_red, 1)), np.tile(BLUE, (n_blue, 1)), np.tile(GREEN, (n_green, 1))]).astype(np.float32)


def test_subview_has_zero_excess_at_any_scale_and_complement_has_full():
    full = P.histogram(person(600, 400), h=100)
    mu = full
    assert P.excess(P.histogram(person(300, 0), 100), mu) == 0.0            # half the red, nothing else
    assert P.excess(P.histogram(person(150, 100), 50), mu) == 0.0           # same person at half the height: counts / 4
    assert P.excess(P.histogram(person(0, 0, 500), 100), mu) == 1.0         # a colour the person never had
    assert abs(P.excess(P.histogram(person(300, 0, 200), 100), mu) - 0.4) < 1e-9  # 2/5 of the pixels unexplained


def test_budget_is_binwise_max_and_covers_each_view():
    b = P.Budget()
    v1, v2 = P.histogram(person(600, 100), 100), P.histogram(person(100, 600), 100)
    b.add(v1)
    b.add(v2)
    assert P.excess(v1, b.mu) == 0.0 and P.excess(v2, b.mu) == 0.0
    assert np.all(b.mu >= v1) and np.all(b.mu >= v2)


def test_conservation_fails_on_doubled_region():
    mu = P.histogram(person(600, 400), 100)
    assert abs(P.conservation(P.histogram(person(600, 400), 100), mu) - 1.0) < 1e-9
    assert P.conservation(P.histogram(person(1200, 800), 100), mu) > 1.5


def test_nonuniform_gain_breaks_chromaticity_and_correction_restores_it():
    pix = person(600, 400)
    mu = P.histogram(pix, 100)
    g = np.array([0.55, 1.0, 1.1])
    dimmed = pix * g
    assert P.excess(P.histogram(dimmed, 100), mu) > 0.5                      # uncorrected: wrong bins
    assert P.excess(P.histogram(dimmed / g, 100), mu) == 0.0                 # corrected: exact
    e, cons, plain = P.excess_certified(dimmed, 100, mu, g * 1.03, np.array([0.05, 0.05, 0.05]))
    assert e == 0.0 and abs(cons - 1.0) < 1e-9


def test_gains_recover_a_diagonal_change():
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 220, size=(256, 320, 3)).astype(np.uint8)
    true_g = np.array([0.7, 0.9, 1.1])
    frame = np.clip(bg * true_g, 0, 255).astype(np.uint8)
    covered = np.zeros((256, 320), bool)
    covered[100:160, 100:200] = True
    g, delta = P.gains(frame, bg, covered)
    assert np.abs(g - true_g).max() < 0.02
    assert delta.max() < 0.05


def test_foreground_excludes_background_nearer_boxes_and_rim():
    bg = np.full((100, 100, 3), 100, np.uint8)
    frame = bg.copy()
    frame[20:80, 30:70] = 200                      # the person
    frame[20:50, 30:50] = 30                       # a nearer person's box region will be cut
    pix, n, area = P.foreground(frame, bg, np.ones(3), (30, 20, 70, 80), nearer=[(30, 20, 50, 50)])
    assert area == 40 * 60
    assert n == (60 - 4) * (40 - 4) - (30 - 2) * (20 - 2)
    assert np.all(pix == 200)


def test_rarity_and_rare_set():
    mu = P.histogram(person(600, 400), 100)
    others = [P.presence(P.histogram(person(500, 0), 100)) for _ in range(9)]      # the crowd is red
    p = P.rarity(others)
    rare = P.rare_set(P.presence(mu), p, p_max=0.1)
    blue_bin = P.bins(BLUE[None])[0]
    red_bin = P.bins(RED[None])[0]
    assert rare[blue_bin] and not rare[red_bin]
    assert P.shows(P.histogram(person(100, 20), 100), rare) >= 1
    assert P.shows(P.histogram(person(100, 0), 100), rare) == 0
