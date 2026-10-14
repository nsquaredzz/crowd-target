"""The colour budget: a one-sided, illumination-corrected, scale-normalised colour certificate.

Objects (THEORY.md Part IV, Theorems 1, 2, 7):
- light meter      per-channel gain g_t of a frame against the static background B, as the median
                   ratio over pixels covered by no person; delta_t its spatial spread (block medians).
                   Under the diagonal (von Kries) model I_t = diag(g_t) I_0 on the background, dividing
                   by g_t restores the reference colours; gamma keeps the model diagonal.
- bins             chromaticity (r, g) = (R, G) / (R + G + B) of the gain-corrected pixel on a K x K
                   grid, plus one dark bin for pixels whose brightest channel is below DARK.
- histogram        counts per bin divided by h^2, h the box height: the area a person covers scales
                   as h^2 under a fixed camera, so these are counts per unit of the person.
- budget mu        bin-wise MAXIMUM over the pre-gap views of the target's per-unit histograms: the
                   most of each colour the person has ever shown.  Any later view made of surfaces
                   already shown is a sub-measure of it.
- excess E         share of a candidate's pixels in excess of (1 + eps) times the one-bin-dilated
                   budget; E <= noise for any sub-view at any scale under any gain in the estimated
                   box (Theorem 1); minimised over the 27-point gain grid, so E > tau certifies "not
                   a sub-view of anything seen, under any illumination in the box".
- conservation     total per-unit mass of a candidate over the budget's; a view is at most as large
                   as the person (Theorem 2).
- presence/rarity  bins holding >= 2 % of a histogram's mass; a target bin is rare when at most
                   p_max of the other people present show its one-bin neighbourhood.  Rare colours
                   confirm; their false rate is the share of the crowd that shows them (Theorem 7).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

K = 24                  # chromaticity bins per axis
DARK = 30.0             # brightest corrected channel below this -> dark bin
NB = K * K + 1


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def background_model(paths, n: int = 60, rows: int = 120) -> np.ndarray:
    """Per-pixel temporal median over n frames spread across the sequence (uint8 H x W x 3)."""
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    frames = [load_rgb(paths[i]) for i in idx]
    H, W, _ = frames[0].shape
    bg = np.empty((H, W, 3), np.uint8)
    for r0 in range(0, H, rows):
        stack = np.stack([f[r0:r0 + rows] for f in frames])
        bg[r0:r0 + rows] = np.median(stack, axis=0).astype(np.uint8)
    return bg


def boxes_mask(boxes, H: int, W: int, pad: int = 4) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4):
        m[max(int(y1) - pad, 0):int(y2) + pad, max(int(x1) - pad, 0):int(x2) + pad] = True
    return m


def gains(frame: np.ndarray, bg: np.ndarray, covered: np.ndarray, block: int = 64, min_bg: int = 20):
    """(g, delta): per-channel gain of `frame` against `bg` over uncovered pixels, and the 90th
    percentile relative deviation of 64-px block gains from it (the spatial non-uniformity)."""
    free = (~covered) & (bg.min(-1) > min_bg)
    f = frame.astype(np.float32)
    b = np.maximum(bg.astype(np.float32), 1.0)
    if free.sum() < 1000:
        return np.ones(3), np.full(3, 0.05)
    g = np.array([float(np.median(f[..., c][free] / b[..., c][free])) for c in range(3)])
    H, W = free.shape
    bl = []
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            v = free[y:y + block, x:x + block]
            if v.sum() >= 0.3 * block * block:
                fb, bb = f[y:y + block, x:x + block], b[y:y + block, x:x + block]
                bl.append([float(np.median(fb[..., c][v] / bb[..., c][v])) for c in range(3)])
    if len(bl) < 4:
        return g, np.full(3, 0.05)
    bl = np.asarray(bl)
    delta = np.quantile(np.abs(bl / g - 1.0), 0.9, axis=0)
    return g, np.clip(delta, 0.02, 0.5)


def bins(J: np.ndarray) -> np.ndarray:
    """Bin index of gain-corrected pixels J[..., 3]."""
    J = np.asarray(J, np.float32)
    mx = J.max(-1)
    s = np.maximum(J.sum(-1), 1e-6)
    br = np.clip((J[..., 0] / s * K).astype(np.int64), 0, K - 1)
    bg_ = np.clip((J[..., 1] / s * K).astype(np.int64), 0, K - 1)
    return np.where(mx < DARK, K * K, br * K + bg_)


def histogram(J: np.ndarray, h: float) -> np.ndarray:
    """Counts per bin per unit h^2 for corrected pixels J[n, 3]."""
    if J.shape[0] == 0:
        return np.zeros(NB)
    return np.bincount(bins(J).ravel(), minlength=NB).astype(np.float64) / float(h) ** 2


def dilate(v: np.ndarray) -> np.ndarray:
    """One-bin dilation on the chromaticity grid; the dark bin is left alone."""
    grid = maximum_filter(np.asarray(v, np.float64)[:K * K].reshape(K, K), size=3, mode="nearest").reshape(-1)
    return np.concatenate([grid, np.asarray(v, np.float64)[K * K:]])


def foreground_mask(frame: np.ndarray, bg: np.ndarray, g: np.ndarray, box, nearer=(), thr: float = 25.0, rim: int = 2):
    """Box-sized bool mask of the person inside `box`: pixels that differ from the gain-corrected
    background by more than thr in some channel, not inside any nearer person's box, not within
    `rim` px of the box edge.  Returns (mask, (x1, y1, x2, y2) clipped) or (None, box) if empty."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    crop = frame[y1:y2, x1:x2].astype(np.float32)
    ref = bg[y1:y2, x1:x2].astype(np.float32) * np.asarray(g, np.float32)
    fg = np.abs(crop - ref).max(-1) > thr
    for nx1, ny1, nx2, ny2 in np.asarray(nearer, np.float64).reshape(-1, 4):
        ax1, ay1 = max(int(nx1), x1) - x1, max(int(ny1), y1) - y1
        ax2, ay2 = min(int(nx2), x2) - x1, min(int(ny2), y2) - y1
        if ax2 > ax1 and ay2 > ay1:
            fg[ay1:ay2, ax1:ax2] = False
    if rim:
        fg[:rim] = False
        fg[-rim:] = False
        fg[:, :rim] = False
        fg[:, -rim:] = False
    return fg, (x1, y1, x2, y2)


def foreground(frame: np.ndarray, bg: np.ndarray, g: np.ndarray, box, nearer=(), thr: float = 25.0, rim: int = 2):
    """Raw (uncorrected) pixels of the person inside `box` (see foreground_mask).
    Returns (pixels[n, 3] float32, n, box area)."""
    fg, (x1, y1, x2, y2) = foreground_mask(frame, bg, g, box, nearer, thr, rim)
    if fg is None:
        return np.zeros((0, 3), np.float32), 0, 0
    crop = frame[y1:y2, x1:x2].astype(np.float32)
    return crop[fg], int(fg.sum()), int((x2 - x1) * (y2 - y1))


class Budget:
    """Bin-wise maximum over views of per-unit histograms."""

    def __init__(self):
        self.frames: list[np.ndarray] = []

    def add(self, hist: np.ndarray) -> None:
        self.frames.append(np.asarray(hist, np.float64))

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def mu(self) -> np.ndarray:
        return np.max(np.stack(self.frames), 0) if self.frames else np.zeros(NB)


def excess(nu: np.ndarray, mu: np.ndarray, eps: float = 0.1) -> float:
    tot = float(nu.sum())
    if tot <= 0:
        return float("nan")
    return float(np.maximum(nu - (1.0 + eps) * dilate(mu), 0.0).sum() / tot)


def conservation(nu: np.ndarray, mu: np.ndarray) -> float:
    return float(nu.sum() / max(float(mu.sum()), 1e-9))


def gain_grid(g: np.ndarray, delta: np.ndarray) -> list[np.ndarray]:
    g, delta = np.asarray(g, np.float64), np.asarray(delta, np.float64)
    return [g * (1.0 + np.array([a, b, c]) * delta) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]


def excess_certified(pixels: np.ndarray, h: float, mu: np.ndarray, g, delta, eps: float = 0.1):
    """(min over the gain grid of E, conservation ratio at that gain, plain E at g)."""
    if pixels.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    plain = excess(histogram(pixels / np.asarray(g, np.float64), h), mu, eps)
    best, cons = np.inf, float("nan")
    for gg in gain_grid(g, delta):
        nu = histogram(pixels / gg, h)
        e = excess(nu, mu, eps)
        if e < best:
            best, cons = e, conservation(nu, mu)
    return float(best), float(cons), float(plain)


# --- presence and rarity (the confirmation channel) -----------------------------------------

def presence(hist: np.ndarray, frac: float = 0.02) -> np.ndarray:
    tot = float(hist.sum())
    if tot <= 0:
        return np.zeros(NB, bool)
    return hist / tot >= frac


def dilate_bool(p: np.ndarray) -> np.ndarray:
    return dilate(p.astype(np.float64)) > 0


def rarity(others_presence) -> np.ndarray:
    """Share of the other people showing each bin's one-bin neighbourhood."""
    if len(others_presence) == 0:
        return np.ones(NB)
    return np.mean([dilate_bool(p) for p in others_presence], axis=0)


def rare_set(target_presence: np.ndarray, p_others: np.ndarray, p_max: float = 0.1) -> np.ndarray:
    return target_presence & (p_others <= p_max)


def shows(hist: np.ndarray, rare: np.ndarray, frac: float = 0.01) -> int:
    """Number of rare bins (with one-bin tolerance) this histogram shows at >= frac of its mass."""
    return int((presence(hist, frac) & dilate_bool(rare)).sum())
