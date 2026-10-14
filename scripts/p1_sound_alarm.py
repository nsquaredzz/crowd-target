#!/usr/bin/env python3
"""P1 redone with the right score: the colour budget's certified excess as the nonconformity stream.

    python scripts/p1_sound_alarm.py --run p0_sam2_bp_dev
    python scripts/p1_sound_alarm.py --run fix_samurai_dev

Same protocol as p1_martingale.py (same martingales, CUSUM and object-score comparators, same
scoring against the P0 outcomes), with the embedding distance replaced by E: the share of SAM 2's
mask pixels (at the stored 1/4 resolution) that the target's own pre-gap budget cannot account for,
minimised over the illumination gain box of the frame.  The pool is the leave-one-out E of the
pre-gap frames on which SAM 2 was confident (object score > 2); the tested stream is every k-th
frame from the gap start on which SAM 2 claims a whole person (score > gate, area >= gate-area x
the pre-gap median).  E is one-sided: a partial view of the same person scores ~0, so unlike the
embedding distance it should not alarm on occlusion or turning unless a new colour appears.
Writes out/p1s_<run>.{json,txt}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from crowdtarget import palette as P  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402
from crowdtarget.martingale import SwitchDetector, cusum_learned_null  # noqa: E402
from p1_martingale import ALPHAS, evaluate  # noqa: E402

DOWN = 4


def mask_boxes_h(m: np.ndarray) -> float:
    ys = np.nonzero(m.any(1))[0]
    return float((ys.max() - ys.min() + 1) * DOWN) if ys.size else float("nan")


def episode_scores(rec, z, bg4, gt, info, seq_dir, k, gate_score=2.0, gate_area=0.5):
    frames = z["frames"]
    masks = np.unpackbits(z["masks_small"], axis=-1)[:, :, : z["mask_shape"][2]].astype(bool)
    h4, w4 = masks.shape[1:]
    area = masks.sum((1, 2)).astype(float)
    scores = z["scores"]
    gap_start = rec["gap_start"]
    pre = [i for i in range(len(frames)) if frames[i] < gap_start and scores[i] > 2.0 and area[i] > 20]
    if len(pre) < 5:
        return None
    pre_area = float(np.median(area[pre]))
    tested = [i for i in range(len(frames)) if frames[i] >= gap_start and scores[i] > gate_score and area[i] >= gate_area * pre_area
              and (frames[i] - gap_start) % k == 0]
    need = sorted(set(pre) | set(tested))
    pix, hh, gg, dd = {}, {}, {}, {}
    for i in need:
        t = int(frames[i])
        f4 = P.load_rgb(frame_path(seq_dir, t, info))[::DOWN, ::DOWN][:h4, :w4]
        rows = gt[gt[:, 0] == t]
        boxes = np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1) / DOWN if len(rows) else np.zeros((0, 4))
        g, d = P.gains(f4, bg4, P.boxes_mask(boxes, h4, w4, pad=1), block=32)
        pix[i] = f4[masks[i]].astype(np.float32)
        hh[i] = mask_boxes_h(masks[i])
        gg[i], dd[i] = g, d
    hists = {i: P.histogram(pix[i] / gg[i], hh[i]) for i in pre}
    # leave-one-out pool
    pool = []
    for j in pre:
        mu_j = np.max(np.stack([hists[i] for i in pre if i != j]), 0)
        pool.append(P.excess(hists[j], mu_j))
    mu = np.max(np.stack([hists[i] for i in pre]), 0)
    s_cert, s_plain = [], []
    for i in tested:
        e, _, e_plain = P.excess_certified(pix[i], hh[i], mu, gg[i], dd[i])
        s_cert.append(e)
        s_plain.append(e_plain)
    out = {"frames_tested": [int(frames[i]) for i in tested], "pool_n": len(pool), "pool": pool, "E_cert": s_cert, "E_plain": s_plain}
    for mk in ("mixture", "jumper"):
        det = SwitchDetector(np.asarray(pool), kind=mk, alpha=1.0, seed=0)
        out[f"wealth_{mk}"] = [det.update(s)[1] for s in s_cert]
    out["cusum"] = cusum_learned_null(np.concatenate([pool, s_cert]), n_cal=len(pool)).tolist()
    post = [i for i in range(len(frames)) if frames[i] >= gap_start]
    out["frames_score"] = [int(frames[i]) for i in post]
    out["obj_score"] = [float(scores[i]) for i in post]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p0_sam2_bp_dev")
    ap.add_argument("--k", nargs="*", type=int, default=[4, 12])
    ap.add_argument("--gate-score", type=float, default=2.0)
    ap.add_argument("--gate-area", type=float, default=0.5)
    a = ap.parse_args()
    payload = json.loads((ROOT / "out" / f"{a.run}.json").read_text())
    records = payload["records"]
    runs = ROOT / "data" / "runs" / a.run
    cache = ROOT / "data" / "cache"
    assets = {}
    t0 = time.time()
    results, lines = {}, [f"P1 with the colour budget's certified excess as the score, run {a.run}, {time.strftime('%Y-%m-%d')}",
                          f"gate: object score > {a.gate_score}, mask area >= {a.gate_area} x pre-gap median; pool = leave-one-out E of confident pre-gap frames", ""]
    for k in a.k:
        alarms = []
        for rec in records:
            seq = rec["seq"]
            if seq not in assets:
                seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / seq
                bg = np.load(cache / f"{seq}_bg.npy")
                assets[seq] = (targets(read_gt(seq_dir)), seq_info(seq_dir), seq_dir, bg[::DOWN, ::DOWN])
            gt, info, seq_dir, bg4 = assets[seq]
            z = np.load(runs / f"{seq}_{rec['id']}_{rec['gap_start']}.npz")
            al = episode_scores(rec, z, bg4[: z["mask_shape"][1], : z["mask_shape"][2]], gt, info, seq_dir, k, a.gate_score, a.gate_area)
            alarms.append(al)
            if al is not None and k == a.k[0]:
                e = np.asarray(al["E_cert"])
                lines.append(f"  {seq}_{rec['id']:<5d} {rec['outcome']:9s} pool E med {np.median(al['pool']):.3f} max {max(al['pool']):.3f} | tested {len(e):3d} frames, "
                             f"E med {np.median(e) if e.size else float('nan'):.3f} max {e.max() if e.size else float('nan'):.3f} | "
                             f"mixture wealth max {max(al['wealth_mixture']) if e.size else 1:.3g}")
        lines.append("")
        lines.append(f"k = {k}: {'detector':22s} {'alpha':>6s} {'detected':>9s} {'false alarms':>13s} {'lost alarmed':>13s} {'delay':>6s}")
        for det_name, thrs in (("mixture", ALPHAS), ("jumper", ALPHAS), ("cusum", (2.0, 4.0, 6.0)), ("objscore", (0.0, 1.0, 2.0))):
            for thr in thrs:
                r = evaluate(records, alarms, det_name, thr)
                results[f"k{k}_{det_name}_{thr}"] = r
                lines.append(f"       {det_name:22s} {thr:6g} {r['detected']:4d} / {r['n_switched']:<3d} {r['false']:6d} / {r['n_recovered']:<5d} "
                             f"{r['lost_alarmed']:6d} / {r['n_lost']:<5d} {r['median_delay']:6.1f}")
        lines.append("")
        np.savez_compressed(ROOT / "data" / "runs" / f"p1s_{a.run}_k{k}.npz", **{f"{rec['seq']}_{rec['id']}": json.dumps(al) for rec, al in zip(records, alarms) if al})
    text = "\n".join(lines) + "\n"
    write_out(f"p1s_{a.run}", {"machine": machine_info(), "results": results, "seconds": round(time.time() - t0, 1), "gate_score": a.gate_score, "gate_area": a.gate_area}, text)
    print(text)


if __name__ == "__main__":
    main()
