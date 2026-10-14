"""Neighbour continuity: whoever was seen the whole time is not the hidden one (Theorem 4).

An IoU tracker on the public detections.  A match is `certain` when the previous box and the
detection overlap at IoU >= iou_match and neither has any other partner at IoU >= iou_amb.  A track
ends at the first frame without a match (strict continuity).  A neighbour is excluded as the hidden
target only if one track covers >= frac of the gap frames with certain matches.  The hypothesis
behind it, that two fully visible people do not swap inside one frame without their boxes meeting,
is measured on ground truth by `rule_violations`.  The census also uses the ground-truth oracle
version (`oracle_visible_frac`) to separate the idea from the detector.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def xywh_to_xyxy(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, np.float64).reshape(-1, 4)
    return np.stack([r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]], 1)


class IouTracker:
    """Tracks: dict id -> {"frames": [t...], "boxes": [xyxy...], "certain": [bool...]}."""

    def __init__(self, iou_match: float = 0.5, iou_amb: float = 0.2):
        self.iou_match, self.iou_amb = iou_match, iou_amb
        self.tracks: dict[int, dict] = {}
        self._active: dict[int, np.ndarray] = {}     # id -> last box
        self._next = 0
        self.last_assignment: dict[int, int] = {}    # detection index -> track id, for the last frame

    def step(self, t: int, dets_xyxy: np.ndarray) -> None:
        D = np.asarray(dets_xyxy, np.float64).reshape(-1, 4)
        ids = list(self._active)
        P = np.stack([self._active[i] for i in ids]) if ids else np.zeros((0, 4))
        M = iou_matrix(P, D)
        assigned_p, assigned_d = set(), set()
        new_active: dict[int, np.ndarray] = {}
        self.last_assignment = {}
        if M.size:
            pairs = np.argwhere(M >= self.iou_match)
            order = np.argsort(-M[pairs[:, 0], pairs[:, 1]])
            amb_p = (M >= self.iou_amb).sum(1)      # partners per previous box
            amb_d = (M >= self.iou_amb).sum(0)      # partners per detection
            for k in order:
                pi, dj = pairs[k]
                if pi in assigned_p or dj in assigned_d:
                    continue
                assigned_p.add(pi)
                assigned_d.add(dj)
                tid = ids[pi]
                certain = bool(amb_p[pi] == 1 and amb_d[dj] == 1)
                tr = self.tracks[tid]
                tr["frames"].append(t)
                tr["boxes"].append(D[dj])
                tr["certain"].append(certain)
                new_active[tid] = D[dj]
                self.last_assignment[int(dj)] = tid
        for dj in range(D.shape[0]):
            if dj in assigned_d:
                continue
            tid = self._next
            self._next += 1
            self.tracks[tid] = {"frames": [t], "boxes": [D[dj]], "certain": [False]}
            new_active[tid] = D[dj]
            self.last_assignment[dj] = tid
        self._active = new_active

    def run(self, dets_by_frame: dict[int, np.ndarray], frames) -> dict[int, dict]:
        for t in frames:
            self.step(int(t), dets_by_frame.get(int(t), np.zeros((0, 4))))
        return self.tracks


def certain_frac(track: dict, t0: int, t1: int) -> float:
    """Share of frames in [t0, t1] on which this track has a certain match."""
    n = t1 - t0 + 1
    if n <= 0:
        return 0.0
    return sum(1 for t, c in zip(track["frames"], track["certain"]) if t0 <= t <= t1 and c) / n


def oracle_visible_frac(gt: np.ndarray, id_: int, t0: int, t1: int, min_vis: float = 0.5) -> float:
    rows = gt[(gt[:, 1] == id_) & (gt[:, 0] >= t0) & (gt[:, 0] <= t1)]
    return float((rows[:, 8] >= min_vis).sum() / max(t1 - t0 + 1, 1))


def gt_id_per_detection(dets_xyxy: np.ndarray, gt_rows: np.ndarray, iou_min: float = 0.5) -> np.ndarray:
    """Ground-truth identity of each detection by best IoU (-1 if below iou_min)."""
    if gt_rows.shape[0] == 0 or dets_xyxy.shape[0] == 0:
        return -np.ones(dets_xyxy.shape[0], int)
    G = xywh_to_xyxy(gt_rows[:, 2:6])
    M = iou_matrix(dets_xyxy, G)
    best = M.argmax(1)
    ok = M[np.arange(M.shape[0]), best] >= iou_min
    return np.where(ok, gt_rows[best, 1].astype(int), -1)


def rule_violations(tracks: dict[int, dict], gt_ids_by_track: dict[int, list[int]], min_run: int = 10, min_each: int = 3) -> dict:
    """Over maximal runs of certain frames of length >= min_run: a run is violated if two or more
    ground-truth identities each appear on >= min_each of its frames."""
    runs = viol = frames = viol_frames = 0
    for tid, tr in tracks.items():
        ids = gt_ids_by_track.get(tid)
        if ids is None:
            continue
        cert = tr["certain"]
        i = 0
        while i < len(cert):
            if not cert[i]:
                i += 1
                continue
            j = i
            while j < len(cert) and cert[j]:
                j += 1
            if j - i >= min_run:
                runs += 1
                frames += j - i
                seg = [g for g in ids[i:j] if g >= 0]
                vals, counts = np.unique(seg, return_counts=True) if seg else (np.array([]), np.array([]))
                if (counts >= min_each).sum() >= 2:
                    viol += 1
                    viol_frames += j - i
            i = j
    return {"runs": runs, "violated_runs": viol, "run_violation_rate": viol / runs if runs else float("nan"),
            "certain_frames": frames, "violated_frames": viol_frames}


def detector_recall(dets_by_frame: dict[int, np.ndarray], gt: np.ndarray, frames, min_vis: float = 0.5, min_h: float = 40, iou_min: float = 0.5) -> dict:
    hit = tot = 0
    for t in frames:
        rows = gt[(gt[:, 0] == t) & (gt[:, 8] >= min_vis) & (gt[:, 5] >= min_h)]
        if rows.shape[0] == 0:
            continue
        D = dets_by_frame.get(int(t), np.zeros((0, 4)))
        tot += rows.shape[0]
        if D.shape[0]:
            M = iou_matrix(xywh_to_xyxy(rows[:, 2:6]), D)
            hit += int((M.max(1) >= iou_min).sum())
    return {"gt_boxes": tot, "detected": hit, "recall": hit / tot if tot else float("nan")}
