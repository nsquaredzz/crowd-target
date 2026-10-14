"""Metric height from a fixed elevated camera over a ground plane (Theorem 5).

For upright people on a plane seen by a fixed camera, pixel height is affine in the image row of
the feet to first order: h = a y + b.  rho = h / (a y + b) is a person's height in units of the
scene's mean height and does not change with depth.  Fitted on fully visible boxes with one pass
of 2-sigma trimming.
"""
from __future__ import annotations

import numpy as np


def height_model(gt: np.ndarray, min_vis: float = 0.9, min_h: float = 40) -> tuple[np.ndarray, float]:
    rows = gt[(gt[:, 8] >= min_vis) & (gt[:, 5] >= min_h)]
    y = rows[:, 3] + rows[:, 5]
    h = rows[:, 5]
    A = np.stack([y, np.ones_like(y)], 1)
    ab, *_ = np.linalg.lstsq(A, h, rcond=None)
    res = h - A @ ab
    keep = np.abs(res) < 2 * res.std()
    ab, *_ = np.linalg.lstsq(A[keep], h[keep], rcond=None)
    rel = (h[keep] - A[keep] @ ab) / (A[keep] @ ab)
    return ab, float(rel.std())


def rho(h, y_foot, ab) -> np.ndarray:
    return np.asarray(h, np.float64) / np.maximum(ab[0] * np.asarray(y_foot, np.float64) + ab[1], 1.0)


def person_rho(gt: np.ndarray, id_: int, ab, frames=None, min_vis: float = 0.5) -> float:
    rows = gt[(gt[:, 1] == id_) & (gt[:, 8] >= min_vis)]
    if frames is not None:
        rows = rows[np.isin(rows[:, 0], np.asarray(list(frames)))]
    if rows.shape[0] == 0:
        return float("nan")
    return float(np.median(rho(rows[:, 5], rows[:, 3] + rows[:, 5], ab)))
