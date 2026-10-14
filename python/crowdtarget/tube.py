"""The occlusion node and its tube: where a hidden person can be (THEORY.md Part IV, Theorem 3).

O_t is the union of every other person's box on a coarse grid (CELL px): wherever a person could be
standing hidden.  From the last visible foot position the reachable set is propagated as
X_t = dilate(X_{t-1}, r) & O_t, r the cells a person can walk in one frame at the speed bound.  The
dilated set before intersection is the reach: a candidate in free space whose foot lies in it is in
the exit set.  Nodes are the connected components of O_t; the occluders involved are the identities whose
boxes hold part of X_t, and a hand-over is a new identity starting to cover X_t.  An empty X_t means the hypotheses failed (the person crossed free space
unseen, or faster than the bound): the reach set is kept as the fallback and the step is counted.
Once the person may be visible again (the gap has ended, or a candidate has appeared) the set is
only dilated (`step_free`): a half-visible person stands in free space and the occluder constraint
no longer applies.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label

CELL = 8
S8 = np.ones((3, 3), bool)


def grid_shape(W: int, H: int, cell: int = CELL) -> tuple[int, int]:
    return (H + cell - 1) // cell, (W + cell - 1) // cell


def occluder_grid(boxes, W: int, H: int, cell: int = CELL) -> np.ndarray:
    gh, gw = grid_shape(W, H, cell)
    O = np.zeros((gh, gw), bool)
    for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4):
        O[max(int(y1 // cell), 0):min(int(np.ceil(y2 / cell)), gh), max(int(x1 // cell), 0):min(int(np.ceil(x2 / cell)), gw)] = True
    return O


def cell_of(x: float, y: float, shape, cell: int = CELL) -> tuple[int, int]:
    iy, ix = int(y // cell), int(x // cell)
    return min(max(iy, 0), shape[0] - 1), min(max(ix, 0), shape[1] - 1)


def cells_per_frame(v_max: float, h: float, cell: int = CELL) -> int:
    return max(int(np.ceil(v_max * h / cell)), 1)


def nodes(O: np.ndarray):
    lab, n = label(O, structure=S8)
    return lab, int(n)


class Tube:
    def __init__(self, W: int, H: int, cell: int = CELL):
        self.shape = grid_shape(W, H, cell)
        self.cell = cell
        self.X = np.zeros(self.shape, bool)
        self.reach = np.zeros(self.shape, bool)
        self.steps = 0
        self.empty_steps = 0
        self.handovers = 0
        self.n_nodes_now = 0
        self.occluders: set[int] = set()
        self._last_cover: set | None = None

    def start(self, foot_xy, radius: int = 1) -> None:
        iy, ix = cell_of(foot_xy[0], foot_xy[1], self.shape, self.cell)
        X = np.zeros(self.shape, bool)
        X[max(iy - radius, 0):iy + radius + 1, max(ix - radius, 0):ix + radius + 1] = True
        self.X, self.reach = X, X.copy()

    def step(self, O: np.ndarray, r_cells: int) -> np.ndarray:
        D = binary_dilation(self.X, S8, iterations=max(int(r_cells), 1))
        self.reach = D
        X = D & O
        self.steps += 1
        if not X.any():
            self.empty_steps += 1
            X = D
        self.X = X
        lab, _ = nodes(O)
        self.n_nodes_now = len(set(int(v) for v in np.unique(lab[X])) - {0})
        return X

    def step_free(self, r_cells: int) -> np.ndarray:
        """The possibly-visible phase: the set grows at the speed bound with no occluder constraint."""
        D = binary_dilation(self.X, S8, iterations=max(int(r_cells), 1))
        self.reach, self.X = D, D
        self.steps += 1
        return D

    def covering(self, boxes, ids) -> set:
        """Identities whose boxes hold some of X: the people the target may be hidden behind now.
        Updates the hand-over count (an id that starts covering X) and the set of occluders involved."""
        now = set()
        for b, i in zip(np.asarray(boxes, np.float64).reshape(-1, 4), ids):
            if occluder_grid(b, self.shape[1] * self.cell, self.shape[0] * self.cell, self.cell)[self.X].any():
                now.add(int(i))
        if self._last_cover is not None and now - self._last_cover:
            self.handovers += 1
        self._last_cover = now
        self.occluders |= now
        return now

    def admits(self, foot_xy) -> bool:
        iy, ix = cell_of(foot_xy[0], foot_xy[1], self.shape, self.cell)
        return bool(self.reach[iy, ix])

    def contains(self, foot_xy) -> bool:
        iy, ix = cell_of(foot_xy[0], foot_xy[1], self.shape, self.cell)
        return bool(self.X[iy, ix])

    @property
    def area_cells(self) -> int:
        return int(self.X.sum())
