"""Companions: neighbours who walked with the target before the gap (coherent-motion grouping).

A companion is another identity whose foot point stayed within `max_dist` body heights of the
target's on at least `min_frames` of the pre-gap frames, with relative velocity below `max_rel_v`
heights per frame.  Used as a ranking cue at exit; the census records the true person's exit
distance to the companion so that a conformal exclusion radius can be judged.
"""
from __future__ import annotations

import numpy as np


def foot(rows: np.ndarray) -> np.ndarray:
    return np.stack([rows[:, 2] + rows[:, 4] / 2, rows[:, 3] + rows[:, 5]], 1)


def companions(gt: np.ndarray, id_: int, t0: int, t1: int, max_dist: float = 1.0, min_frames: int = 20, max_rel_v: float = 0.01) -> list[dict]:
    tgt = gt[(gt[:, 1] == id_) & (gt[:, 0] >= t0) & (gt[:, 0] <= t1)]
    if tgt.shape[0] < min_frames:
        return []
    tgt = tgt[np.argsort(tgt[:, 0])]
    h = float(np.median(tgt[:, 5]))
    ft = {int(r[0]): f for r, f in zip(tgt, foot(tgt))}
    out = []
    win = gt[(gt[:, 0] >= t0) & (gt[:, 0] <= t1) & (gt[:, 1] != id_)]
    for j in np.unique(win[:, 1]).astype(int):
        rows = win[win[:, 1] == j]
        rows = rows[np.argsort(rows[:, 0])]
        fj = foot(rows)
        common = [(int(r[0]), f) for r, f in zip(rows, fj) if int(r[0]) in ft]
        if len(common) < min_frames:
            continue
        d = np.array([np.hypot(*(f - ft[t])) for t, f in common]) / h
        if (d <= max_dist).sum() < min_frames:
            continue
        ts = np.array([t for t, _ in common])
        rel = np.array([f - ft[t] for t, f in common])
        if len(ts) >= 2:
            v = np.hypot(*np.diff(rel, axis=0).T) / np.maximum(np.diff(ts), 1) / h
            if np.median(v) > max_rel_v:
                continue
        out.append({"id": int(j), "median_dist": float(np.median(d)), "frames": int(len(common))})
    return sorted(out, key=lambda c: c["median_dist"])
