"""Scoring a single-target episode against MOTChallenge ground truth.

Per frame the tracker's mask is compared with the ground-truth boxes of that frame by
containment, the share of the mask's pixels inside a box (masks are tighter than boxes, so IoU
would punish a correct mask).  Outcomes per frame: correct (containment in the target's box
>= thr), switched:<id> (containment in another identity's box >= thr), absent (empty mask),
wrong (a mask nowhere).  Per episode:

  switched   a run of >= run_len consecutive frames on the same wrong identity, anywhere after
             the prompt frame; the first such run gives the id, the frame and the phase
             (gap: inside the occlusion; post: after re-emergence)
  recovered  not switched, and correct on >= frac_ok of the post frames where the ground truth
             is visible (visibility >= vis_ok)
  lost       otherwise
  delay      first post frame starting a run of 5 correct frames, counted from re-emergence
"""
from __future__ import annotations

import numpy as np

from .data.mot import rows_at, targets


def containment(mask: np.ndarray, box_xyxy) -> float:
    n = int(mask.sum())
    if n == 0:
        return 0.0
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    H, W = mask.shape
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].sum() / n)


def frame_outcome(mask: np.ndarray, gt: np.ndarray, t: int, target_id: int, thr: float = 0.5,
                  scale: float = 1.0) -> str:
    """mask at 1/scale of the GT resolution (scale = gt_px / mask_px)."""
    if mask.sum() == 0:
        return "absent"
    rows = rows_at(targets(gt), t)
    best_other, best_other_id = 0.0, None
    target_c = 0.0
    for r in rows:
        box = np.array([r[2], r[3], r[2] + r[4], r[3] + r[5]]) / scale
        c = containment(mask, box)
        if int(r[1]) == target_id:
            target_c = c
        elif c > best_other:
            best_other, best_other_id = c, int(r[1])
    if target_c >= thr:
        return "correct"
    if best_other >= thr:
        return f"switched:{best_other_id}"
    return "wrong"


def episode_outcome(per_frame: list[str], frames: list[int], gap_start: int, gap_end: int,
                    gt_visible_post: dict[int, bool], run_len: int = 10, frac_ok: float = 0.5) -> dict:
    out = {"outcome": None, "switch_id": None, "switch_frame": None, "switch_phase": None, "delay": None}
    # switched runs
    run_id, run_n, run_t0 = None, 0, None
    for t, o in zip(frames, per_frame):
        if o.startswith("switched:"):
            sid = int(o.split(":")[1])
            if sid == run_id:
                run_n += 1
            else:
                run_id, run_n, run_t0 = sid, 1, t
            if run_n >= run_len and out["switch_frame"] is None:
                out["switch_id"], out["switch_frame"] = sid, run_t0
                out["switch_phase"] = "gap" if gap_start <= run_t0 <= gap_end else ("post" if run_t0 > gap_end else "pre")
        else:
            run_id, run_n, run_t0 = None, 0, None
    post = [(t, o) for t, o in zip(frames, per_frame) if t > gap_end and gt_visible_post.get(t, False)]
    n_ok = sum(o == "correct" for _, o in post)
    frac = n_ok / len(post) if post else 0.0
    out["post_correct_frac"] = float(frac)
    out["n_post_visible"] = len(post)
    if out["switch_frame"] is not None:
        out["outcome"] = "switched"
    elif frac >= frac_ok:
        out["outcome"] = "recovered"
    else:
        out["outcome"] = "lost"
    streak = 0
    for t, o in post:
        streak = streak + 1 if o == "correct" else 0
        if streak >= 5:
            out["delay"] = int(t - 4 - gap_end)
            break
    return out


def summarise(records: list[dict]) -> dict:
    """Counts by bin and overall: recovered / switched (gap, post) / lost, median delay."""
    out = {}
    bins = sorted({r["bin"] for r in records}) + ["all"]
    for b in bins:
        rs = [r for r in records if b == "all" or r["bin"] == b]
        n = len(rs)
        d = {"n": n}
        for k in ("recovered", "switched", "lost", "ambiguous"):
            d[k] = sum(r["outcome"] == k for r in rs)
        d["switched_gap"] = sum(r["outcome"] == "switched" and r["switch_phase"] == "gap" for r in rs)
        d["switched_post"] = sum(r["outcome"] == "switched" and r["switch_phase"] == "post" for r in rs)
        delays = [r["delay"] for r in rs if r.get("delay") is not None]
        d["median_delay"] = float(np.median(delays)) if delays else float("nan")
        out[b] = d
    return out


def format_summary(s: dict, title: str) -> str:
    lines = [title, "", f"{'bin':8s} {'n':>4s} {'recov':>6s} {'switch':>7s} {'(gap':>5s} {'post)':>6s} {'lost':>5s} {'ambig':>6s} {'delay':>6s}"]
    for b, d in s.items():
        lines.append(f"{b:8s} {d['n']:4d} {d['recovered']:6d} {d['switched']:7d} {d['switched_gap']:5d} {d['switched_post']:6d} "
                     f"{d['lost']:5d} {d.get('ambiguous', 0):6d} {d['median_delay']:6.1f}")
    return "\n".join(lines) + "\n"
