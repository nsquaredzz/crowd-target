"""Certified texture anchors: fine detail that confirms (layer 7, Theorem 7's second channel).

A template is an s x s patch of the gain-corrected luminance on the person whose self-match set is
{0}: no shift of two pixels or more certifies against it (certified-track Theorem P), and every
scale of the box test has at least one feature (kappa >= 1).  At exit a candidate's crop is
resampled to the anchor's scale (h_src / h_cand) and searched: a mean-removed SAD at stride 2
proposes positions, the multi-scale box test with the exact brightness offset decides.  Score < 1
is the certificate.  Rarity: an anchor is kept only if it certifies on none of the other people in
its frame; the crowd pass rate is recorded for Theorem 7.  Resampling is interpolation, so the
exactness of the raw-pixel certificate is lost; what remains is the measured false rate.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from certtrack.track import certify

S = 21                       # template side
HALF = S // 2
SELF_R = 6                   # self-match radius
DELTAS = certify.schedule(80.0)


def luminance(frame_rgb: np.ndarray, g) -> np.ndarray:
    return (frame_rgb.astype(np.float32) / np.asarray(g, np.float32)).mean(-1)


def mine(lum: np.ndarray, box, fg_mask: np.ndarray | None, stride: int = 6, max_n: int = 8, erode: int = 3) -> list[dict]:
    """Anchors inside `box` of frame luminance `lum`; fg_mask (box-sized bool) restricts centres to the person."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    H, W = lum.shape
    h_src = float(y2 - y1)
    out = []
    R = SELF_R + HALF
    if fg_mask is not None and erode:
        fg_mask = binary_erosion(fg_mask, iterations=erode)
    for cy in range(y1 + R + 1, y2 - R - 1, stride):
        for cx in range(x1 + R + 1, x2 - R - 1, stride):
            if cy - R < 0 or cx - R < 0 or cy + R + 1 > H or cx + R + 1 > W:
                continue
            if fg_mask is not None and not fg_mask[cy - y1, cx - x1]:
                continue
            win = lum[cy - R:cy + R + 1, cx - R:cx + R + 1].astype(np.float64)
            T = win[SELF_R:SELF_R + S, SELF_R:SELF_R + S]
            if T.std() < 8:
                continue
            kap = certify.kappa(T, DELTAS)
            if (kap < 1).any():
                continue
            sm = certify.selfmatch(win, S, DELTAS)
            n_self, margin = certify.selfmatch_stats(sm)
            if n_self:
                continue
            out.append({"T": T.copy(), "margin": float(margin), "u": (cx - x1) / max(x2 - x1, 1), "v": (cy - y1) / h_src,
                        "h_src": h_src, "kappa": kap.tolist()})
    out.sort(key=lambda a: -a["margin"])
    return out[:max_n]


def resampled_crop(lum: np.ndarray, box, h_src: float, margin: float = 0.15) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if h < 20 or w < 8:
        return None
    H, W = lum.shape
    X1, Y1 = max(int(x1 - margin * w), 0), max(int(y1 - margin * h), 0)
    X2, Y2 = min(int(x2 + margin * w), W), min(int(y2 + margin * h), H)
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return None
    f = h_src / h
    im = Image.fromarray(lum[Y1:Y2, X1:X2].astype(np.float32), mode="F")
    size = (max(int(round((X2 - X1) * f)), S + 2), max(int(round((Y2 - Y1) * f)), S + 2))
    return np.asarray(im.resize(size, Image.BILINEAR), np.float64)


def best_score(crop: np.ndarray, T: np.ndarray, top: int = 5) -> float:
    """Minimum certificate score of template T over the crop: SAD proposals at stride 2, box test at the best."""
    if crop.shape[0] < S or crop.shape[1] < S:
        return float("inf")
    win = np.lib.stride_tricks.sliding_window_view(crop, (S, S))[::2, ::2]
    Tm = T - T.mean()
    wm = win - win.mean((-2, -1), keepdims=True)
    sad = np.abs(wm - Tm).sum((-2, -1))
    flat = np.argsort(sad.ravel())[:top]
    ys, xs = np.unravel_index(flat, sad.shape)
    cands = []
    for y, x in zip(ys * 2, xs * 2):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy <= crop.shape[0] - S and 0 <= xx <= crop.shape[1] - S:
                    cands.append(crop[yy:yy + S, xx:xx + S])
    if not cands:
        return float("inf")
    sc, _ = certify.score(np.stack(cands), T[None], DELTAS, "opt")
    return float(sc.min())


def certified_count(lum: np.ndarray, box, anchors: list[dict]) -> int:
    """How many anchors certify (score < 1) somewhere on the candidate's resampled crop."""
    if not anchors:
        return 0
    n = 0
    for a in anchors:
        crop = resampled_crop(lum, box, a["h_src"])
        if crop is None:
            continue
        if best_score(crop, a["T"]) < 1.0:
            n += 1
    return n


def keep_distinctive(anchors: list[dict], lum: np.ndarray, other_boxes) -> tuple[list[dict], int]:
    """Drop anchors that certify on any other person's crop in this frame; returns (kept, n_others)."""
    others = [b for b in np.asarray(other_boxes, np.float64).reshape(-1, 4)]
    kept = []
    for a in anchors:
        hit = False
        for b in others:
            crop = resampled_crop(lum, b, a["h_src"])
            if crop is not None and best_score(crop, a["T"]) < 1.0:
                hit = True
                break
        a["crowd_hits"] = int(hit)
        if not hit:
            kept.append(a)
    return kept, len(others)
