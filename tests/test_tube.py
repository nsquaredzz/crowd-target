import numpy as np

from crowdtarget import tube as T


def box(x1, y1, x2, y2):
    return np.array([[x1, y1, x2, y2]], float)


def test_handover_follows_the_second_occluder_and_exit_is_on_its_side():
    W, H = 800, 400
    A = box(80, 100, 144, 210)       # cells 10..17
    B = box(130, 100, 200, 210)      # cells 16..24, overlaps A
    tb = T.Tube(W, H)
    tb.start((100, 205))             # last seen foot under A
    r = T.cells_per_frame(0.045, 110)
    assert r == 1
    for t in range(1, 4):
        tb.step(T.occluder_grid(np.concatenate([A, B]), W, H), r)
    for t in range(4, 12):           # A gone, only B remains
        tb.step(T.occluder_grid(B, W, H), r)
    assert tb.empty_steps == 0
    ys, xs = np.nonzero(tb.X)
    assert xs.min() >= 130 // 8 and xs.max() < int(np.ceil(200 / 8))     # inside B's columns
    assert tb.admits((197, 220)) and not tb.admits((400, 220)) and not tb.admits((60, 220))


def test_free_space_crossing_empties_the_tube_and_is_flagged():
    W, H = 800, 400
    A = box(80, 100, 144, 210)
    B = box(400, 100, 460, 210)      # far away, never touching
    tb = T.Tube(W, H)
    tb.start((100, 205))
    r = 1
    for t in range(3):
        tb.step(T.occluder_grid(np.concatenate([A, B]), W, H), r)
    assert tb.empty_steps == 0
    tb.step(T.occluder_grid(B, W, H), r)           # A vanished: nowhere to hide within reach
    assert tb.empty_steps == 1
    assert tb.X.any()                               # fallback keeps the reach set


def test_covering_ids_record_the_handover():
    W, H = 800, 400
    A = box(80, 100, 144, 210)
    C = box(145, 100, 220, 210)      # cell-adjacent to A
    tb = T.Tube(W, H)
    tb.start((100, 205))
    for _ in range(6):               # X fills A
        tb.step(T.occluder_grid(A, W, H), 1)
        assert tb.covering(A, [7]) == {7}
    assert tb.handovers == 0 and tb.occluders == {7}
    tb.step(T.occluder_grid(np.concatenate([A, C]), W, H), 1)
    assert tb.covering(np.concatenate([A, C]), [7, 9]) == {7, 9}      # X spilled into C
    assert tb.handovers == 1
    tb.step(T.occluder_grid(C, W, H), 1)                             # A walked away
    assert tb.covering(C, [9]) == {9} and tb.empty_steps == 0
    assert tb.handovers == 1 and tb.occluders == {7, 9} and tb.n_nodes_now == 1


def test_step_free_grows_without_occluders():
    tb = T.Tube(800, 400)
    tb.start((100, 205))
    assert not tb.admits((140, 205))
    for _ in range(4):
        tb.step_free(1)
    assert tb.admits((140, 205)) and tb.empty_steps == 0
