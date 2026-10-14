#!/usr/bin/env python3
"""Census 2: is the "hidden" person really hidden?  Three ground-truth-only measurements on the 120 episodes.

    python scripts/census2.py

1. Slivers.  On every gap frame (visibility < 0.3), the pixels inside the target's box that lie in no
   nearer person's box and differ from the static background by more than 25 in some channel: the
   part of the person that still shows.  Reported: share of gap frames with >= 100 / >= 300 such
   pixels, and the longest run of gap frames with fewer than 100 (the true blackout).
2. Heads.  The top 15 % of the target's box, middle half in width, is the head region; it is covered
   when >= 50 % of it lies inside nearer boxes.  Reported: share of gap frames with the head
   uncovered, and the longest covered run.  (Elevated camera: heads occlude each other less than bodies.)
3. Depth order.  A person hidden behind others stands farther from the camera than they do: their
   foot row is above the covering occluders' foot rows.  At exit, a candidate whose foot row is below
   the frontmost covering occluder's by more than the speed bound allows cannot be the hidden person.
   Reported: share of reach-disc impostors excluded, and true persons wrongly excluded.
Writes out/census2.{json,txt}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from census import feet, group_by_frame, speed_bound, xyxy  # noqa: E402
from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import palette as P  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402

CACHE = ROOT / "data" / "cache"


def longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def head_region(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + 0.25 * w, y1, x2 - 0.25 * w, y1 + 0.15 * h])


def covered_frac(region, boxes):
    """Share of `region`'s area inside the union of `boxes` (grid approximation)."""
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return 1.0
    xs = np.linspace(x1, x2, 12)
    ys = np.linspace(y1, y2, 12)
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros(gx.shape, bool)
    for bx1, by1, bx2, by2 in np.asarray(boxes).reshape(-1, 4):
        inside |= (gx >= bx1) & (gx <= bx2) & (gy >= by1) & (gy <= by2)
    return float(inside.mean())


def main():
    eps = E.read(ROOT / "data" / "episodes" / "mot20_all.json")
    assets = {}
    recs = []
    t0 = time.time()
    for k, ep in enumerate(eps):
        if ep.seq not in assets:
            seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
            gt = targets(read_gt(seq_dir))
            assets[ep.seq] = (seq_dir, gt, group_by_frame(gt), seq_info(seq_dir), np.load(CACHE / f"{ep.seq}_bg.npy"), speed_bound(gt))
        seq_dir, gt, bf, info, bg, vmax = assets[ep.seq]
        tgt = gt[gt[:, 1] == ep.id]
        pre = tgt[(tgt[:, 0] < ep.gap_start) & (tgt[:, 0] >= ep.t_start)]
        h_ref = float(np.median(pre[:, 5])) if len(pre) else float(tgt[0, 5])
        sliv, heads, front_y2, resid = [], [], [], []
        last_pre = tgt[tgt[:, 0] < ep.gap_start]
        last_pre = last_pre[np.argmax(last_pre[:, 0])]
        for t in range(ep.gap_start, ep.gap_end + 1):
            rows = bf.get(t)
            if rows is None:
                continue
            r = rows[rows[:, 1] == ep.id]
            if len(r) == 0:
                continue
            r = r[0]
            box = xyxy(r)[0]
            others = rows[rows[:, 1] != ep.id]
            nearer = xyxy(others[(others[:, 3] + others[:, 5]) > (r[3] + r[5])]) if len(others) else np.zeros((0, 4))
            frame = P.load_rgb(frame_path(seq_dir, t, info))
            pix, n, area = P.foreground(frame, bg, np.ones(3), box, nearer)
            sliv.append(n)
            # 4. the residual without the target's own box: unexplained foreground near the last position,
            #    all other people's boxes removed; is its largest blob on the target?
            dt = t - ep.gap_start + 1
            reach = vmax * dt * h_ref + 0.5 * h_ref
            lx1, ly1, lx2, ly2 = xyxy(last_pre)[0]
            rx1, ry1 = int(max(lx1 - reach, 0)), int(max(ly1 - reach, 0))
            rx2, ry2 = int(min(lx2 + reach, info["W"])), int(min(ly2 + reach, info["H"]))
            crop = frame[ry1:ry2, rx1:rx2].astype(np.float32)
            fg = np.abs(crop - bg[ry1:ry2, rx1:rx2].astype(np.float32)).max(-1) > 25
            for ox1, oy1, ox2, oy2 in xyxy(others):
                ax1, ay1 = max(int(ox1), rx1) - rx1, max(int(oy1), ry1) - ry1
                ax2, ay2 = min(int(ox2), rx2) - rx1, min(int(oy2), ry2) - ry1
                if ax2 > ax1 and ay2 > ay1:
                    fg[ay1:ay2, ax1:ax2] = False
            from scipy.ndimage import label as _label
            lab, nl = _label(fg)
            if nl:
                sizes = np.bincount(lab.ravel())[1:]
                j = int(np.argmax(sizes)) + 1
                if sizes[j - 1] >= 100:
                    ys, xs = np.nonzero(lab == j)
                    cy, cx = ys.mean() + ry1, xs.mean() + rx1
                    resid.append((box[0] <= cx <= box[2] and box[1] <= cy <= box[3], int(sizes[j - 1]), np.hypot(cx - (box[0] + box[2]) / 2, cy - (box[1] + box[3]) / 2) / h_ref))
                else:
                    resid.append((False, int(sizes[j - 1]), float("nan")))
            else:
                resid.append((False, 0, float("nan")))
            heads.append(covered_frac(head_region(box), nearer) < 0.5)
            # covering occluders: nearer boxes intersecting the target's box
            cov = [b for b in nearer if min(b[2], box[2]) > max(b[0], box[0]) and min(b[3], box[3]) > max(b[1], box[1])]
            if cov:
                front_y2.append(min(b[3] for b in cov))           # the nearest (lowest) covering occluder's foot row... smallest y2 among coverers = the farthest coverer
        if not sliv:
            continue
        sliv = np.asarray(sliv)
        # depth order at exit: candidates in the reach disc
        post = tgt[(tgt[:, 0] > ep.gap_end) & (tgt[:, 0] <= ep.gap_end + 20) & (tgt[:, 8] >= 0.5)]
        post = post[np.argsort(post[:, 0])][:5]
        last = tgt[tgt[:, 0] < ep.gap_start]
        last = last[np.argmax(last[:, 0])]
        foot0 = feet(last)[0]
        disc_r = vmax * (ep.g + 10) * h_ref
        margin = vmax * (ep.g + 10) * h_ref                   # rows the person could have walked towards the camera
        y2_limit = (max(front_y2) + margin) if front_y2 else None    # a hidden person stood above every coverer's feet... use the max over frames of the nearest coverer
        n_imp = n_imp_excl = 0
        true_excl = None
        for t in post[:, 0].astype(int):
            rows = bf.get(t)
            vis = rows[(rows[:, 8] >= 0.5) & (rows[:, 5] >= 40)]
            for o, fo in zip(vis, feet(vis)):
                if np.hypot(*(fo - foot0)) > disc_r:
                    continue
                excl = (y2_limit is not None) and (o[3] + o[5] > y2_limit)
                if int(o[1]) == ep.id:
                    true_excl = bool(excl) if true_excl is None else (true_excl and bool(excl))
                else:
                    n_imp += 1
                    n_imp_excl += int(excl)
        recs.append({"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "g": ep.g, "bin": ep.bin, "n_gap": int(len(sliv)),
                     "sliver_ge100": float(np.mean(sliv >= 100)), "sliver_ge300": float(np.mean(sliv >= 300)),
                     "sliver_median": float(np.median(sliv)), "blackout_100": int(longest_run(sliv < 100)),
                     "head_free": float(np.mean(heads)), "head_blackout": int(longest_run([not x for x in heads])),
                     "depth_imp": n_imp, "depth_imp_excl": n_imp_excl, "depth_true_excl": true_excl,
                     "resid_on_target": float(np.mean([a for a, _, _ in resid])) if resid else float("nan"),
                     "resid_on_target_when_big": float(np.mean([a for a, n_, _ in resid if n_ >= 300])) if any(n_ >= 300 for _, n_, _ in resid) else float("nan"),
                     "resid_frames_big": float(np.mean([n_ >= 300 for _, n_, _ in resid])) if resid else float("nan")})
        if (k + 1) % 10 == 0:
            print(f"[{k + 1}/{len(eps)}] {time.time() - t0:.0f} s", flush=True)
    R = recs

    def blk(rs):
        return {"n": len(rs), "gap_frames_with_slivers_ge100": float(np.mean([r["sliver_ge100"] for r in rs])),
                "gap_frames_with_slivers_ge300": float(np.mean([r["sliver_ge300"] for r in rs])),
                "sliver_px_median": float(np.median([r["sliver_median"] for r in rs])),
                "blackout_median": float(np.median([r["blackout_100"] for r in rs])), "blackout_q90": float(np.quantile([r["blackout_100"] for r in rs], 0.9)),
                "blackout_le5": float(np.mean([r["blackout_100"] <= 5 for r in rs])), "blackout_le10": float(np.mean([r["blackout_100"] <= 10 for r in rs])),
                "head_free_frames": float(np.mean([r["head_free"] for r in rs])), "head_blackout_median": float(np.median([r["head_blackout"] for r in rs])),
                "head_blackout_le5": float(np.mean([r["head_blackout"] <= 5 for r in rs])),
                "depth_excluded_impostors": float(sum(r["depth_imp_excl"] for r in rs) / max(sum(r["depth_imp"] for r in rs), 1)),
                "depth_true_wrongly_excluded": float(np.mean([bool(r["depth_true_excl"]) for r in rs if r["depth_true_excl"] is not None])),
                "resid_on_target": float(np.nanmean([r["resid_on_target"] for r in rs])),
                "resid_on_target_when_big": float(np.nanmean([r["resid_on_target_when_big"] for r in rs])),
                "resid_frames_big": float(np.nanmean([r["resid_frames_big"] for r in rs]))}

    S = {"all": blk(R)}
    for b in sorted({r["bin"] for r in R}):
        S[b] = blk([r for r in R if r["bin"] == b])
    for s in sorted({r["seq"] for r in R}):
        S[s] = blk([r for r in R if r["seq"] == s])
    L = [f"Census 2: how hidden is the hidden person?  {len(R)} MOT20 episodes, ground truth only, {time.strftime('%Y-%m-%d')}", "",
         f"{'group':9s} {'n':>3s} | {'sliv>=100':>9s} {'sliv>=300':>9s} {'px med':>7s} {'blackout med':>12s} {'q90':>5s} {'<=5':>5s} {'<=10':>5s} | {'head free':>9s} {'head bo med':>11s} {'<=5':>5s} | {'depth excl':>10s} {'true excl':>9s} | {'resid big':>9s} {'on tgt':>6s} {'on tgt|big':>10s}"]
    for k, b in S.items():
        L.append(f"{k:9s} {b['n']:3d} | {b['gap_frames_with_slivers_ge100']:9.2f} {b['gap_frames_with_slivers_ge300']:9.2f} {b['sliver_px_median']:7.0f} {b['blackout_median']:12.0f} {b['blackout_q90']:5.0f} {b['blackout_le5']:5.2f} {b['blackout_le10']:5.2f} | "
                 f"{b['head_free_frames']:9.2f} {b['head_blackout_median']:11.0f} {b['head_blackout_le5']:5.2f} | {b['depth_excluded_impostors']:10.2f} {b['depth_true_wrongly_excluded']:9.2f} | {b['resid_frames_big']:9.2f} {b['resid_on_target']:6.2f} {b['resid_on_target_when_big']:10.2f}")
    L += ["", "sliv>=100: share of gap frames (visibility < 0.3) on which >= 100 px of the person still show outside nearer boxes;",
          "blackout: longest run of gap frames with < 100 px showing (median, 90th percentile, share of episodes with a blackout of",
          "<= 5 / <= 10 frames); head free: share of gap frames with the head region < 50 % covered; depth excl: share of",
          "reach-disc impostors at exit standing nearer than the frontmost coverer allows; true excl: true person so excluded;",
          "resid: the largest blob of unexplained foreground (frame minus background minus every other person's box) within reach",
          "of the last position: share of gap frames where it has >= 300 px, and share where its centroid falls in the target's box", ""]
    for r in R:
        L.append(f"{r['seq']}_{r['id']}_{r['gap_start']:<6d} g {r['g']:3d}  gap frames {r['n_gap']:3d}  slivers>=100 {100 * r['sliver_ge100']:3.0f} %  px med {r['sliver_median']:5.0f}  blackout {r['blackout_100']:3d}  "
                 f"head free {100 * r['head_free']:3.0f} %  head blackout {r['head_blackout']:3d}  depth excl {r['depth_imp_excl']}/{r['depth_imp']} true-excl {r['depth_true_excl']}")
    text = "\n".join(L) + "\n"
    write_out("census2", {"machine": machine_info(), "records": R, "summary": S, "seconds": round(time.time() - t0, 1)}, text)
    print("\n".join(L[:len(S) + 9]))


if __name__ == "__main__":
    main()
