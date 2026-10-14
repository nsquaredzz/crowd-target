#!/usr/bin/env python3
"""Census 4: multi-template certified sliver chains through the gaps, ground truth only.

    python scripts/census_chain.py [--min-sliver 100]

Per gap frame the sliver mask is the person's foreground inside its box and outside nearer boxes.  A
Chain (crowdtarget.sliver) is seeded on the first gap frame with >= min-sliver px and advanced frame
by frame: certified positions must fall inside the target's box at the new frame (on target) or
outside it (off target: a false step).  When the sliver vanishes the search widens at walking speed;
a certified return after k blank frames is a bridged blackout.  Reported: share of frame pairs with a
usable template set, certified steps on target, false steps, chain coverage of the gap, bridged
blackouts by length.  Kill criteria of the plan: usable >= 70 %, false < 1 %.  Writes out/census_chain.{json,txt}.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from census import group_by_frame, xyxy  # noqa: E402
from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import palette as P  # noqa: E402
from crowdtarget import sliver as SL  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402

CACHE = ROOT / "data" / "cache"


def sliver_full(frame, bg, rows, r):
    """Full-frame bool mask of the target's visible sliver (box ∩ foreground, nearer boxes removed)."""
    box = xyxy(r)[0]
    others = rows[rows[:, 1] != r[1]]
    nearer = xyxy(others[(others[:, 3] + others[:, 5]) > (r[3] + r[5])]) if len(others) else np.zeros((0, 4))
    fg, (x1, y1, x2, y2) = P.foreground_mask(frame, bg, np.ones(3), box, nearer)
    full = np.zeros(frame.shape[:2], bool)
    if fg is not None:
        full[y1:y2, x1:x2] = fg
    return full, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sliver", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delta0", type=float, default=80.0, help="certificate contrast Delta_0 (80 default, 128 looser)")
    a = ap.parse_args()
    SL.set_delta0(a.delta0)
    stem = "census_chain" + ("" if a.delta0 == 80.0 else f"_d{int(a.delta0)}")
    eps = E.read(ROOT / "data" / "episodes" / "mot20_all.json")[: a.limit]
    assets = {}
    recs = []
    t0 = time.time()
    for k, ep in enumerate(eps):
        if ep.seq not in assets:
            seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
            gt = targets(read_gt(seq_dir))
            assets[ep.seq] = (seq_dir, gt, group_by_frame(gt), seq_info(seq_dir), np.load(CACHE / f"{ep.seq}_bg.npy"))
        seq_dir, gt, bf, info, bg = assets[ep.seq]
        chain = SL.Chain()
        seeded = False
        pairs = usable = on = off = off_near = 0
        covered = []            # per gap frame: joined to the previous by a certified step
        bridged = {}            # blackout length -> [attempts, successes]
        blank = 0
        prev_box = None
        for t in range(ep.gap_start, ep.gap_end + 1):
            rows = bf.get(t)
            r = rows[rows[:, 1] == ep.id] if rows is not None else []
            if len(r) == 0:
                covered.append(False)
                blank += 1
                continue
            r = r[0]
            frame = P.load_rgb(frame_path(seq_dir, t, info))
            lum = SL.luminance(frame)
            mask, box = sliver_full(frame, bg, rows, r)
            n = int(mask.sum())
            if not seeded:
                if n >= a.min_sliver and chain.seed(lum, mask):
                    seeded = True
                covered.append(False)
                continue
            pairs += 1
            if chain.templates:
                usable += 1
            found = chain.advance(lum, mask if n >= a.min_sliver else None)
            if found:
                hits = sum(1 for f in found if box[0] <= f["x"] <= box[2] and box[1] <= f["y"] <= box[3])
                on += hits
                off += len(found) - hits
                for f in found:
                    if not (box[0] <= f["x"] <= box[2] and box[1] <= f["y"] <= box[3]):
                        dist = max(box[0] - f["x"], f["x"] - box[2], box[1] - f["y"], f["y"] - box[3])
                        off_near += dist <= 5
                if blank:
                    bridged.setdefault(blank, [0, 0])
                    bridged[blank][0] += 1
                    bridged[blank][1] += 1
                covered.append(True)
                blank = 0
            else:
                covered.append(False)
                if n >= a.min_sliver and blank:
                    bridged.setdefault(blank, [0, 0])
                    bridged[blank][0] += 1
                blank = blank + 1 if n < a.min_sliver else blank
                if n >= a.min_sliver and chain.templates == []:
                    chain.seed(lum, mask)          # reseed on a visible sliver when nothing is left to follow
            if n >= a.min_sliver and not found and chain.broken_for >= 8:
                chain.seed(lum, mask)              # give up on the old templates after a long blank
            prev_box = box
        cov = np.asarray(covered, bool)
        best = cur = 0
        for f in cov:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        recs.append({"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "g": ep.g, "bin": ep.bin, "seeded": seeded,
                     "pairs": pairs, "usable": usable, "on": on, "off": off, "off_near": off_near, "covered_frac": float(cov.mean()) if cov.size else 0.0,
                     "longest_chain": best, "bridged": {str(b): v for b, v in bridged.items()}})
        if (k + 1) % 10 == 0:
            print(f"[{k + 1}/{len(eps)}] {time.time() - t0:.0f} s", flush=True)

    def blk(rs):
        pairs = sum(r["pairs"] for r in rs)
        usable = sum(r["usable"] for r in rs)
        on = sum(r["on"] for r in rs)
        off = sum(r["off"] for r in rs)
        off_near = sum(r.get("off_near", 0) for r in rs)
        br = {}
        for r in rs:
            for b, (att, suc) in r["bridged"].items():
                br.setdefault(int(b), [0, 0])
                br[int(b)][0] += att
                br[int(b)][1] += suc
        return {"n": len(rs), "seeded": float(np.mean([r["seeded"] for r in rs])), "pairs": pairs, "usable_frac": usable / max(pairs, 1),
                "positions_on_target": on, "positions_off_target": off, "false_frac": off / max(on + off, 1),
                "off_within_5px_of_box": off_near / max(off, 1),
                "covered_median": float(np.median([r["covered_frac"] for r in rs])), "covered_mean": float(np.mean([r["covered_frac"] for r in rs])),
                "longest_chain_median": float(np.median([r["longest_chain"] for r in rs])),
                "longest_over_gap_median": float(np.median([r["longest_chain"] / max(r["g"], 1) for r in rs])),
                "bridged": {b: {"attempts": v[0], "bridged": v[1]} for b, v in sorted(br.items())}}

    R = recs
    Sm = {"all": blk(R)}
    for b in sorted({r["bin"] for r in R}):
        Sm[b] = blk([r for r in R if r["bin"] == b])
    for s in sorted({r["seq"] for r in R}):
        Sm[s] = blk([r for r in R if r["seq"] == s])
    L = [f"Census 4: multi-template certified sliver chains through the gaps, {len(R)} episodes, min sliver {a.min_sliver} px, Delta_0 {a.delta0:g}, {time.strftime('%Y-%m-%d')}", "",
         f"{'group':9s} {'n':>3s} {'seeded':>6s} {'pairs':>6s} {'usable':>6s} {'on tgt':>7s} {'off tgt':>7s} {'false':>6s} {'off<=5px':>8s} {'covered med':>11s} {'mean':>5s} {'chain med':>9s} {'chain/gap':>9s}"]
    for k_, b in Sm.items():
        L.append(f"{k_:9s} {b['n']:3d} {b['seeded']:6.2f} {b['pairs']:6d} {b['usable_frac']:6.2f} {b['positions_on_target']:7d} {b['positions_off_target']:7d} {b['false_frac']:6.3f} {b['off_within_5px_of_box']:8.2f} "
                 f"{b['covered_median']:11.2f} {b['covered_mean']:5.2f} {b['longest_chain_median']:9.0f} {b['longest_over_gap_median']:9.2f}")
    br = Sm["all"]["bridged"]
    L += ["", "Blackouts bridged (a certified return after k frames without a sliver): " +
          (", ".join(f"k={b}: {v['bridged']}/{v['attempts']}" for b, v in br.items()) if br else "none attempted"),
          "", "usable: a template set existed at the previous frame; on/off tgt: certified template positions inside / outside the target's",
          "box on the new frame (off = false identity step); covered: share of gap frames joined to the previous by a certified step;",
          "chain: longest run of consecutive certified steps, absolute and over the gap length."]
    text = "\n".join(L) + "\n"
    write_out(stem, {"machine": machine_info(), "records": R, "summary": Sm, "min_sliver": a.min_sliver, "delta0": a.delta0, "seconds": round(time.time() - t0, 1)}, text)
    print(text)


if __name__ == "__main__":
    main()
