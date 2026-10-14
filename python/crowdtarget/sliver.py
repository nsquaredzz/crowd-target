"""Certified sliver chains: identity continuity for a person who is one-sixth visible.

A sliver is the part of the person that still shows while they are "occluded".  Templates are 21 x 21
luminance patches cut inside the sliver whose self-match set is {0} (certified-track Theorem P: no
shift of 2 px or more certifies against them) with at least one feature at every scale of the box
test (kappa >= 1).  A *step* searches each template within +-r px on the next frame and certifies the
best position with the multi-scale box test and the exact brightness offset (certified-skip Theorem 3
with certified-track's exact offset): score < 1 means no feature of contrast Delta_r appeared or
vanished in the patch up to one brightness constant.  A chain is a run of frames joined by at least
one certified step.  Census 3 measured one template per frame: certified on the target 82 % of the
time it existed, on the nearest neighbour 0 of 1,288.  Several templates per sliver are what make the
chain dense enough to carry identity through a gap.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, label

from certtrack.track import certify

S = 21
HALF = S // 2
SELF_R = 6
WIN_R = SELF_R + HALF
DELTAS = certify.schedule(80.0)


def set_delta0(delta0: float) -> None:
    """Contrast schedule of the certificate: Delta_r = delta0 (2r+1)^-1/2; 80 is the certified-skip default."""
    DELTAS.clear()
    DELTAS.update(certify.schedule(float(delta0)))


def luminance(frame_rgb: np.ndarray, g=None) -> np.ndarray:
    f = frame_rgb.astype(np.float32)
    if g is not None:
        f = f / np.asarray(g, np.float32)
    return f.mean(-1)


def largest_component(mask: np.ndarray, min_px: int = 100):
    lab, n = label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    j = int(np.argmax(sizes)) + 1
    if sizes[j - 1] < min_px:
        return None
    return lab == j


def mine(lum: np.ndarray, mask: np.ndarray, stride: int = 4, max_n: int = 8, erode: int = HALF, min_sep: int = 8,
         min_std: float = 8.0, max_cands: int = 80, seed: int = 0) -> list[dict]:
    """Templates inside a full-frame bool `mask`: grid centres on the mask eroded by `erode` px (HALF by default,
    so the whole 21 x 21 patch lies on the person and never on the occluder next to it), at most `max_cands`
    sampled, kept if S = {0} and kappa >= 1 at every scale, the best `max_n` by self-match margin, >= `min_sep` apart."""
    H, W = lum.shape
    m = binary_erosion(mask, iterations=erode) if erode else mask
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    grid = [(cy, cx) for cy in range(int(y0), int(y1) + 1, stride) for cx in range(int(x0), int(x1) + 1, stride)
            if m[cy, cx] and cy - WIN_R >= 0 and cx - WIN_R >= 0 and cy + WIN_R + 1 <= H and cx + WIN_R + 1 <= W]
    if len(grid) > max_cands:
        idx = np.random.default_rng(seed).choice(len(grid), max_cands, replace=False)
        grid = [grid[i] for i in sorted(idx)]
    cands = []
    for cy, cx in grid:
        if True:
            win = lum[cy - WIN_R:cy + WIN_R + 1, cx - WIN_R:cx + WIN_R + 1].astype(np.float64)
            T = win[SELF_R:SELF_R + S, SELF_R:SELF_R + S]
            if T.std() < min_std:
                continue
            if (certify.kappa(T, DELTAS) < 1).any():
                continue
            n_self, margin = certify.selfmatch_stats(certify.selfmatch(win, S, DELTAS))
            if n_self:
                continue
            cands.append({"T": T.copy(), "y": int(cy), "x": int(cx), "margin": float(margin)})
    cands.sort(key=lambda a: -a["margin"])
    out = []
    for c in cands:
        if all(max(abs(c["y"] - o["y"]), abs(c["x"] - o["x"])) >= min_sep for o in out):
            out.append(c)
        if len(out) >= max_n:
            break
    return out


def best_position(crop: np.ndarray, T: np.ndarray, top: int = 5):
    """(min certificate score, dy, dx) of T over `crop`, offsets of the window's top-left; SAD proposals at stride 1."""
    if crop.shape[0] < S or crop.shape[1] < S:
        return float("inf"), 0, 0
    win = np.lib.stride_tricks.sliding_window_view(crop, (S, S))
    Tm = T - T.mean()
    sad = np.abs((win - win.mean((-2, -1), keepdims=True)) - Tm).sum((-2, -1))
    flat = np.argsort(sad.ravel())[:top]
    ys, xs = np.unravel_index(flat, sad.shape)
    stack = np.stack([crop[y:y + S, x:x + S] for y, x in zip(ys, xs)])
    sc, _ = certify.score(stack, T[None], DELTAS, "opt")
    k = int(np.argmin(sc))
    return float(sc[k]), int(ys[k]), int(xs[k])


def step(lum_next: np.ndarray, templates: list[dict], search_r: int = 10, pred=(0, 0)) -> list[dict]:
    """Certified positions of the templates on the next frame: [{T, y, x, score, from}] with score < 1."""
    H, W = lum_next.shape
    out = []
    for t in templates:
        cy, cx = t["y"] + int(pred[0]), t["x"] + int(pred[1])
        y0, x0 = cy - HALF - search_r, cx - HALF - search_r
        y1, x1 = cy + HALF + search_r + 1, cx + HALF + search_r + 1
        yy0, xx0 = max(y0, 0), max(x0, 0)
        crop = lum_next[yy0:min(y1, H), xx0:min(x1, W)]
        sc, dy, dx = best_position(crop, t["T"])
        if sc < 1.0:
            out.append({"T": t["T"], "y": yy0 + dy + HALF, "x": xx0 + dx + HALF, "score": sc, "from": (t["y"], t["x"])})
    return out


def centre(positions: list[dict]):
    if not positions:
        return None
    return float(np.median([p["y"] for p in positions])), float(np.median([p["x"] for p in positions]))


class Chain:
    """The running template set of one person's sliver and how long it has been certified."""

    def __init__(self, max_n: int = 8, search_r: int = 10):
        self.templates: list[dict] = []
        self.max_n, self.search_r = max_n, search_r
        self.length = 0                 # consecutive certified steps
        self.broken_for = 0             # frames since the last certified step
        self.last_pos = None            # (y, x) of the last certified sliver centre

    def seed(self, lum: np.ndarray, mask: np.ndarray) -> int:
        self.templates = mine(lum, mask, max_n=self.max_n)
        if self.templates:
            self.last_pos = centre(self.templates)
        return len(self.templates)

    def probe(self, lum_next: np.ndarray, pred=(0, 0)) -> list[dict]:
        """Certified positions of the current templates on a new frame; no state change.  A blackout
        widens the search at walking speed."""
        r = self.search_r + 7 * self.broken_for
        return step(lum_next, self.templates, min(r, 60), pred)

    def commit(self, found: list[dict], lum: np.ndarray, mask: np.ndarray | None = None) -> None:
        """Accept a step: certified templates are kept at their new positions, fresh ones mined from `mask`."""
        self.length += 1
        self.broken_for = 0
        self.last_pos = centre(found) if found else self.last_pos
        if mask is not None:                       # a certified template that left the person's mask is dropped
            H, W = mask.shape
            found = [f for f in found if 0 <= f["y"] < H and 0 <= f["x"] < W and mask[f["y"], f["x"]]]
        kept = [{"T": f["T"], "y": f["y"], "x": f["x"], "margin": 0.0} for f in found]
        if mask is not None:
            for c in mine(lum, mask, max_n=self.max_n):
                if len(kept) >= self.max_n:
                    break
                if all(max(abs(c["y"] - o["y"]), abs(c["x"] - o["x"])) >= 8 for o in kept):
                    kept.append(c)
        if kept:
            self.templates = kept

    def miss(self) -> None:
        self.length = 0
        self.broken_for += 1

    def advance(self, lum_next: np.ndarray, mask_next: np.ndarray | None = None, pred=(0, 0)) -> list[dict]:
        """probe, then commit or miss (the census loop)."""
        found = self.probe(lum_next, pred)
        if found:
            self.commit(found, lum_next, mask_next)
        else:
            self.miss()
        return found

    def inside(self, found: list[dict], mask: np.ndarray, frac: float = 0.5) -> bool:
        """Do at least `frac` of the certified positions fall on `mask`?"""
        if not found:
            return False
        H, W = mask.shape
        hits = sum(1 for f in found if 0 <= f["y"] < H and 0 <= f["x"] < W and mask[f["y"], f["x"]])
        return hits >= max(1, int(np.ceil(frac * len(found))))
