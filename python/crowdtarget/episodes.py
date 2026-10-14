"""Single-target episodes mined from MOTChallenge ground truth: a long occlusion gap with a clean
run before and after, so that a tracker started before the gap can be scored after it.

Rules (fixed 2026-10-15, before any run):
  gap          maximal run of frames where the identity has visibility < vis_gap or no row
  g            gap length, g_min <= g <= g_max
  pre window   the `pre` frames before the gap: at least `pre_ok` of them at visibility >= vis_ok,
               and the first of them at visibility >= vis_start
  post window  the `post` frames after the gap: at least `post_ok` at visibility >= vis_ok
  size         box height >= min_h at the start of the pre window and at re-emergence
  per identity at most `max_per_id` episodes (the earliest)
  bins         20-40, 40-80, 80+ by g
  dev split    sort by (seq, id, gap_start); within each bin take every k-th with
               k = floor(n / per_bin); the rest is held out
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data.mot import targets, track

BINS = ((20, 40), (40, 80), (80, 10 ** 9))


def bin_of(g: int) -> str:
    for lo, hi in BINS:
        if lo <= g < hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    return "out"


@dataclass
class Episode:
    seq: str
    id: int
    t_start: int        # first frame of the pre window (1-based), the tracker is prompted here
    gap_start: int
    gap_end: int        # last frame of the gap (inclusive)
    t_end: int          # last frame scored
    g: int
    bin: str
    start_box: list     # x1, y1, x2, y2 at t_start
    reemerge_box: list  # x1, y1, x2, y2 at gap_end + 1
    W: int
    H: int
    fps: float

    @property
    def n_frames(self) -> int:
        return self.t_end - self.t_start + 1


def visibility_series(gt: np.ndarray, id_: int, n_frames: int) -> np.ndarray:
    """(n_frames + 1,) visibility per frame (index = frame number), 0 where the identity has no row."""
    v = np.zeros(n_frames + 1)
    r = track(targets(gt), id_)
    v[r[:, 0].astype(int)] = r[:, 8]
    return v


def _box_at(gt: np.ndarray, id_: int, t: int):
    r = gt[(gt[:, 1] == id_) & (gt[:, 0] == t)]
    if r.shape[0] == 0:
        return None
    x, y, w, h = r[0, 2:6]
    return [float(x), float(y), float(x + w), float(y + h)]


def mine(gt: np.ndarray, seq: str, n_frames: int, W: int, H: int, fps: float,
         g_min: int = 20, g_max: int = 200, pre: int = 30, post: int = 60,
         vis_gap: float = 0.2, vis_ok: float = 0.5, vis_start: float = 0.7,
         pre_ok: int = 20, post_ok: int = 50, min_h: float = 48.0, max_per_id: int = 2) -> list[Episode]:
    eps: list[Episode] = []
    ids = np.unique(targets(gt)[:, 1]).astype(int)
    for id_ in ids:
        v = visibility_series(gt, id_, n_frames)
        present = v >= vis_gap
        # the identity's own span: first and last frame with any row
        rows = track(targets(gt), id_)
        f0, f1 = int(rows[0, 0]), int(rows[-1, 0])
        t = f0
        found = 0
        while t <= f1 and found < max_per_id:
            if present[t]:
                t += 1
                continue
            gs = t
            while t <= f1 and not present[t]:
                t += 1
            ge = t - 1
            g = ge - gs + 1
            if not (g_min <= g <= g_max):
                continue
            ts, te = gs - pre, ge + post
            if ts < max(f0, 1) or te > n_frames:
                continue
            pre_v = v[ts:gs]
            post_v = v[ge + 1:te + 1]
            if (pre_v >= vis_ok).sum() < pre_ok or v[ts] < vis_start or (post_v >= vis_ok).sum() < post_ok:
                continue
            b0, b1 = _box_at(gt, id_, ts), _box_at(gt, id_, ge + 1)
            if b0 is None or b1 is None or (b0[3] - b0[1]) < min_h or (b1[3] - b1[1]) < min_h:
                continue
            eps.append(Episode(seq, int(id_), ts, gs, ge, te, g, bin_of(g), b0, b1, W, H, fps))
            found += 1
    return eps


def dev_split(eps: list[Episode], per_bin: int = 8) -> tuple[list[Episode], list[Episode]]:
    eps = sorted(eps, key=lambda e: (e.seq, e.id, e.gap_start))
    dev, held = [], []
    for b in sorted({e.bin for e in eps}):
        group = [e for e in eps if e.bin == b]
        k = max(len(group) // per_bin, 1)
        chosen = group[::k][:per_bin]
        ids = {(e.seq, e.id, e.gap_start) for e in chosen}
        dev += chosen
        held += [e for e in group if (e.seq, e.id, e.gap_start) not in ids]
    return dev, held


def write(eps: list[Episode], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(e) for e in eps], indent=1))


def read(path: Path) -> list[Episode]:
    return [Episode(**d) for d in json.loads(Path(path).read_text())]


def summary(eps: list[Episode]) -> dict:
    out = {"n": len(eps)}
    for b in [f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" for lo, hi in BINS]:
        out[b] = sum(e.bin == b for e in eps)
    out["seqs"] = sorted({e.seq for e in eps})
    return out
