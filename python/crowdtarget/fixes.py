"""Published training-free fixes for SAM 2's identity drift, re-implemented on top of sam2io.

Both work by changing what SAM 2.1 remembers and which of its three candidate masks it keeps;
the network is untouched.  Neither is our code: they follow the papers, with the deviations
listed, so that the comparison in RESULTS.md is against what is available today.

A. SAMURAI (Yang et al., 2024, arXiv 2411.11922), "motion-aware memory".
   - A Kalman filter with a constant-velocity model on the target box.
   - Among the three candidate masks the one kept maximises
         alpha_kf * IoU(candidate box, Kalman prediction) + (1 - alpha_kf) * affinity,
     affinity being SAM 2's own IoU prediction for the candidate; alpha_kf = 0.15.
   - A frame enters the memory bank only if affinity > 0.5, object score > 0 and the Kalman
     IoU > 0.15; the bank holds the N_max most recent qualifying frames.  SAM 2 reads its bank
     by frame index (frame_idx - 1 .. frame_idx - 6), so before each step the qualifying frames
     are re-aliased to those six indices; the temporal position encodings then follow rank,
     which is what SAMURAI does.
   Deviation: the paper's Kalman state is (x, y, aspect, h) after SORT; here it is
   (cx, cy, w, h), which gives the same IoU test.

B. DAM4SAM-style distractor anchors (Videnovic, Lukezic, Kristan, 2024, arXiv 2411.17576),
   approximated from the paper, not their code.
   - Recent memory is updated only on frames where SAM 2 judges the object present
     (object score > 0); absent frames are dropped from it, so the bank does not fill with
     the occluder during a gap.
   - A distractor is present on a frame when a non-chosen candidate mask covers a region at
     least `distractor_frac` of the chosen mask's area outside the chosen mask.  Such a frame,
     if the chosen mask is confident (object score > 2, affinity > 0.7), is promoted to a
     conditioning frame (DAM4SAM's distractor-resolving memory); at most `n_anchors` anchors
     are kept, oldest dropped.
   Deviation: DAM4SAM also scores anchors by distance to existing ones and updates its recent
   memory on a schedule; neither is reproduced.

The patch point is `sam_mask_decoder.forward`: it returns the candidate logits and their
affinities before SAM 2 takes the argmax, so returning the affinities with the chosen index
boosted makes SAM 2 keep that candidate (mask, memory and object pointer alike).
"""
from __future__ import annotations

import numpy as np


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, w, h); the SORT model with w, h in place of a, h."""

    def __init__(self, box_xyxy, dt: float = 1.0, std_pos: float = 1 / 20, std_vel: float = 1 / 160):
        self.std_pos, self.std_vel = std_pos, std_vel
        z = self._z(box_xyxy)
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        self.P = np.diag(np.r_[(2 * std_pos * h) ** 2 * np.ones(4), (10 * std_vel * h) ** 2 * np.ones(4)])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4) * dt
        self.H = np.eye(4, 8)

    @staticmethod
    def _z(b):
        b = np.asarray(b, np.float64)
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]])

    def predict(self) -> np.ndarray:
        h = max(self.x[3], 1.0)
        q = np.r_[(self.std_pos * h) ** 2 * np.ones(4), (self.std_vel * h) ** 2 * np.ones(4)]
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q)
        return self.box

    def update(self, box_xyxy) -> None:
        h = max(self.x[3], 1.0)
        R = np.diag((self.std_pos * h) ** 2 * np.ones(4))
        z = self._z(box_xyxy)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Fix:
    """Base: installs the decoder hook and exposes per-frame diagnostics in `self.last`."""
    name = "none"

    def __init__(self, pred, W: int, H: int):
        self.pred, self.W, self.H = pred, W, H
        self.last: dict = {}
        self.active = False
        self._installed = False

    # --- the hook ---------------------------------------------------------------------------
    def install(self) -> None:
        if self._installed:
            return
        dec = self.pred.sam_mask_decoder
        orig = dec.forward
        fix = self

        def forward(*args, **kw):
            low, ious, toks, obj = orig(*args, **kw)
            if fix.active and low.shape[1] > 1:
                ious = fix._select(low, ious, obj)
            return low, ious, toks, obj

        dec.forward = forward
        dec.crowdtarget_orig_forward = orig
        self._installed = True

    def uninstall(self) -> None:
        dec = self.pred.sam_mask_decoder
        if hasattr(dec, "crowdtarget_orig_forward"):
            dec.forward = dec.crowdtarget_orig_forward
            del dec.crowdtarget_orig_forward
        self._installed = False

    def _candidates(self, low):
        """Candidate masks (M, h, w) bool and their boxes in frame pixels."""
        m = (low[0] > 0).cpu().numpy()
        h, w = m.shape[-2:]
        sx, sy = self.W / w, self.H / h
        boxes = []
        for k in range(m.shape[0]):
            b = mask_box(m[k])
            boxes.append(None if b is None else b * np.array([sx, sy, sx, sy]))
        return m, boxes

    def _select(self, low, ious, obj):
        """Return `ious` with the chosen candidate boosted so SAM 2's argmax picks it."""
        aff = ious[0].float().cpu().numpy()
        masks, boxes = self._candidates(low)
        score = float(obj.reshape(-1)[0])
        k = self.choose(aff, boxes, masks, score)
        self.last = {"chosen": int(k), "affinity": float(aff[k]), "aff_all": aff.tolist(), "obj_score": score,
                     "boxes": boxes, "masks": masks}
        out = ious.clone()
        out[0, k] = out[0, k] + 10.0
        return out

    # --- to override -------------------------------------------------------------------------
    def choose(self, aff, boxes, masks, score) -> int:
        return int(np.argmax(aff))

    def before_step(self, state, i: int) -> None:
        self.active = True

    def after_step(self, state, i: int, mask: np.ndarray, score: float) -> dict:
        self.active = False
        return {}


class Samurai(Fix):
    name = "samurai"

    def __init__(self, pred, W, H, start_box, alpha_kf=0.15, tau_mask=0.5, tau_obj=0.0, tau_kf=0.15, n_max=6):
        super().__init__(pred, W, H)
        self.kf = KalmanBox(start_box)
        self.alpha_kf, self.tau_mask, self.tau_obj, self.tau_kf, self.n_max = alpha_kf, tau_mask, tau_obj, tau_kf, n_max
        self.bank: list[tuple[int, dict]] = []     # (frame, output) of qualifying frames
        self.pred_box = None

    def choose(self, aff, boxes, masks, score) -> int:
        kf = np.array([box_iou(b, self.pred_box) for b in boxes])
        s = self.alpha_kf * kf + (1 - self.alpha_kf) * aff
        k = int(np.argmax(s))
        self.last_kf_all = kf
        return k

    def before_step(self, state, i: int) -> None:
        self.active = True
        self.pred_box = self.kf.predict()
        d = state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        d.clear()
        for r, (_, out) in enumerate(reversed(self.bank[-self.n_max:]), 1):
            d[i - r] = out            # re-alias: rank r qualifying frame at index i - r

    def after_step(self, state, i: int, mask: np.ndarray, score: float) -> dict:
        self.active = False
        d = state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        box = mask_box(mask)
        kf_iou = box_iou(box, self.pred_box)
        aff = self.last.get("affinity", float("nan"))
        ok = (aff > self.tau_mask) and (score > self.tau_obj) and (kf_iou > self.tau_kf)
        if ok and i in d:
            self.bank.append((i, d[i]))
            self.bank = self.bank[-self.n_max:]
            self.kf.update(box)
        return {"kf_iou": kf_iou, "affinity": aff, "in_memory": bool(ok), "pred_box": self.pred_box.tolist(),
                "chosen": self.last.get("chosen", -1)}


class DamAnchors(Fix):
    name = "dam"

    def __init__(self, pred, W, H, start_box, distractor_frac=0.3, tau_obj_anchor=2.0, tau_aff_anchor=0.7, n_anchors=3):
        super().__init__(pred, W, H)
        self.distractor_frac, self.tau_obj_anchor, self.tau_aff_anchor, self.n_anchors = distractor_frac, tau_obj_anchor, tau_aff_anchor, n_anchors
        self.anchors: list[int] = []

    def distractor(self) -> float:
        """Largest area of a non-chosen candidate outside the chosen mask, relative to the chosen mask."""
        masks = self.last.get("masks")
        if masks is None:
            return 0.0
        k = self.last["chosen"]
        area = masks[k].sum()
        if area == 0:
            return 0.0
        return float(max((masks[j] & ~masks[k]).sum() for j in range(masks.shape[0]) if j != k) / area)

    def before_step(self, state, i: int) -> None:
        self.active = True

    def after_step(self, state, i: int, mask: np.ndarray, score: float) -> dict:
        self.active = False
        d = state["output_dict_per_obj"][0]
        dis = self.distractor()
        aff = self.last.get("affinity", float("nan"))
        anchored = False
        if score <= 0:
            d["non_cond_frame_outputs"].pop(i, None)          # absent: do not remember the occluder
        elif dis >= self.distractor_frac and score > self.tau_obj_anchor and aff > self.tau_aff_anchor:
            if i in d["non_cond_frame_outputs"]:
                d["cond_frame_outputs"][i] = d["non_cond_frame_outputs"].pop(i)
                self.anchors.append(i)
                anchored = True
                while len(self.anchors) > self.n_anchors:
                    old = self.anchors.pop(0)
                    d["cond_frame_outputs"].pop(old, None)
        return {"distractor": dis, "affinity": aff, "anchored": anchored, "chosen": self.last.get("chosen", -1)}


def make_fix(kind: str, pred, W: int, H: int, start_box) -> Fix | None:
    if kind in (None, "none", "baseline"):
        return None
    if kind == "samurai":
        return Samurai(pred, W, H, start_box)
    if kind == "dam":
        return DamAnchors(pred, W, H, start_box)
    raise ValueError(kind)
