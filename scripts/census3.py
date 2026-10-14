#!/usr/bin/env python3
"""Census 3: can the visible sliver of a "hidden" person be followed frame to frame with a certificate?

    python scripts/census3.py

On consecutive gap frames (t, t+1) where the target shows >= 300 sliver pixels (inside its box, outside
nearer boxes, differing from the background), a 21 x 21 luminance template is cut at the centroid of
the largest sliver component at t, kept only if its self-match set is {0} (certified-track Theorem P:
no shift of 2+ px certifies against it).  At t+1 it is searched within +-10 px and certified with the
multi-scale box test and the exact brightness offset (score < 1).  Control: the same template searched
at t+1 around the nearest other visible person (false certification).  Reported: share of pairs with a
usable template, share certified on the target, share certified on the neighbour, and the longest run
of consecutive gap frames bridged by certified steps per episode.  Writes out/census3.{json,txt}.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from census import feet, group_by_frame, xyxy  # noqa: E402
from certtrack.track import certify  # noqa: E402
from crowdtarget import anchors as AN  # noqa: E402
from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import palette as P  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402

CACHE = ROOT / "data" / "cache"
S, HALF, R_SEARCH, SELF_R = 21, 10, 10, 6
MIN_SLIVER = 300


def sliver_mask(frame, bg, rows, r):
    box = xyxy(r)[0]
    others = rows[rows[:, 1] != r[1]]
    nearer = xyxy(others[(others[:, 3] + others[:, 5]) > (r[3] + r[5])]) if len(others) else np.zeros((0, 4))
    fg, (x1, y1, x2, y2) = P.foreground_mask(frame, bg, np.ones(3), box, nearer)
    return fg, (x1, y1, x2, y2), others


def main():
    eps = E.read(ROOT / "data" / "episodes" / "mot20_all.json")
    assets = {}
    recs = []
    t0 = time.time()
    deltas = certify.schedule(80.0)
    for k, ep in enumerate(eps):
        if ep.seq not in assets:
            seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
            gt = targets(read_gt(seq_dir))
            assets[ep.seq] = (seq_dir, gt, group_by_frame(gt), seq_info(seq_dir), np.load(CACHE / f"{ep.seq}_bg.npy"))
        seq_dir, gt, bf, info, bg = assets[ep.seq]
        prev = None
        pairs = usable = cert_t = cert_o = 0
        chain = []
        for t in range(ep.gap_start, ep.gap_end + 2):
            rows = bf.get(t)
            r = rows[rows[:, 1] == ep.id] if rows is not None else []
            if len(r) == 0:
                prev = None
                chain.append(False)
                continue
            r = r[0]
            frame = P.load_rgb(frame_path(seq_dir, t, info))
            lum = frame.astype(np.float32).mean(-1)
            fg, (x1, y1, x2, y2), others = sliver_mask(frame, bg, rows, r)
            n = int(fg.sum()) if fg is not None else 0
            if prev is not None and prev["n"] >= MIN_SLIVER and n >= MIN_SLIVER:
                pairs += 1
                T, (cy, cx) = prev["T"], prev["c"]
                if T is not None:
                    usable += 1
                    y0, y1_ = int(cy - HALF - R_SEARCH), int(cy + HALF + R_SEARCH + 1)
                    x0, x1_ = int(cx - HALF - R_SEARCH), int(cx + HALF + R_SEARCH + 1)
                    crop = lum[max(y0, 0):y1_, max(x0, 0):x1_]
                    ok_t = AN.best_score(crop, T, top=8) < 1.0 if crop.shape[0] >= S and crop.shape[1] >= S else False
                    cert_t += ok_t
                    chain.append(bool(ok_t))
                    # control: the nearest other visible person's centre at t
                    ov = others[(others[:, 8] >= 0.5) & (others[:, 5] >= 40)]
                    if len(ov):
                        fo = feet(ov)
                        j = int(np.argmin(np.hypot(*(fo - np.array([cx, cy])).T)))
                        o = ov[j]
                        ocx, ocy = o[2] + o[4] / 2, o[3] + o[5] / 2
                        oy0, oy1 = int(ocy - HALF - R_SEARCH), int(ocy + HALF + R_SEARCH + 1)
                        ox0, ox1 = int(ocx - HALF - R_SEARCH), int(ocx + HALF + R_SEARCH + 1)
                        ocrop = lum[max(oy0, 0):oy1, max(ox0, 0):ox1]
                        cert_o += AN.best_score(ocrop, T, top=8) < 1.0 if ocrop.shape[0] >= S and ocrop.shape[1] >= S else False
                else:
                    chain.append(False)
            else:
                chain.append(False)
            # template for the next step: centroid of the largest sliver component, if it allows a 21 px patch
            T = None
            c = (0.0, 0.0)
            if n >= MIN_SLIVER:
                lab, nl = label(fg)
                sizes = np.bincount(lab.ravel())[1:]
                j = int(np.argmax(sizes)) + 1
                ys, xs = np.nonzero(lab == j)
                cy, cx = ys.mean() + y1, xs.mean() + x1
                Rw = SELF_R + HALF
                if cy - Rw >= 0 and cx - Rw >= 0 and cy + Rw + 1 <= lum.shape[0] and cx + Rw + 1 <= lum.shape[1]:
                    win = lum[int(cy) - Rw:int(cy) + Rw + 1, int(cx) - Rw:int(cx) + Rw + 1].astype(np.float64)
                    Tc = win[SELF_R:SELF_R + S, SELF_R:SELF_R + S]
                    if Tc.std() >= 8 and (certify.kappa(Tc, deltas) >= 1).all():
                        sm = certify.selfmatch(win, S, deltas)
                        n_self, _ = certify.selfmatch_stats(sm)
                        if n_self == 0:
                            T = Tc.copy()
                            c = (int(cy), int(cx))
            prev = {"n": n, "T": T, "c": c}
        best = cur = 0
        for f in chain:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        recs.append({"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "g": ep.g, "bin": ep.bin, "pairs": pairs, "usable": usable,
                     "cert_target": cert_t, "cert_other": cert_o, "longest_chain": best, "gap_frames": ep.g})
        if (k + 1) % 10 == 0:
            print(f"[{k + 1}/{len(eps)}] {time.time() - t0:.0f} s", flush=True)

    def blk(rs):
        pairs = sum(r["pairs"] for r in rs)
        usable = sum(r["usable"] for r in rs)
        return {"n": len(rs), "pairs": pairs, "usable_frac": usable / max(pairs, 1), "cert_target_frac": sum(r["cert_target"] for r in rs) / max(usable, 1),
                "cert_other_frac": sum(r["cert_other"] for r in rs) / max(usable, 1),
                "chain_median": float(np.median([r["longest_chain"] for r in rs])), "chain_over_gap_median": float(np.median([r["longest_chain"] / max(r["gap_frames"], 1) for r in rs]))}

    R = recs
    Sm = {"all": blk(R)}
    for b in sorted({r["bin"] for r in R}):
        Sm[b] = blk([r for r in R if r["bin"] == b])
    for s in sorted({r["seq"] for r in R}):
        Sm[s] = blk([r for r in R if r["seq"] == s])
    L = [f"Census 3: certified sliver steps through the gap, {len(R)} episodes, {time.strftime('%Y-%m-%d')}", "",
         f"{'group':9s} {'n':>3s} {'pairs':>6s} {'usable':>6s} {'cert tgt':>8s} {'cert other':>10s} {'chain med':>9s} {'chain/gap':>9s}"]
    for k, b in Sm.items():
        L.append(f"{k:9s} {b['n']:3d} {b['pairs']:6d} {b['usable_frac']:6.2f} {b['cert_target_frac']:8.2f} {b['cert_other_frac']:10.2f} {b['chain_median']:9.0f} {b['chain_over_gap_median']:9.2f}")
    L += ["", "pairs: consecutive gap frames with >= 300 sliver px on both; usable: a 21 px template with S = {0} and kappa >= 1 at the",
          "sliver's centroid; cert tgt: the template certifies within +-10 px on the next frame; cert other: it certifies around the",
          "nearest other person (false certification); chain: longest run of consecutive certified steps, absolute and as a share of the gap."]
    text = "\n".join(L) + "\n"
    write_out("census3", {"machine": machine_info(), "records": R, "summary": Sm, "seconds": round(time.time() - t0, 1)}, text)
    print(text)


if __name__ == "__main__":
    main()
