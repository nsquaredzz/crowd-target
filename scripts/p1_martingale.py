#!/usr/bin/env python3
"""P1: the conformal test martingale as an identity-switch detector, on the P0 logs.

    python scripts/p1_martingale.py --run p0_sam2_bp_dev
    python scripts/p1_martingale.py --run p0_sam2_bp_dev --k 1 4 8 12 --score centroid nn

Per episode the nonconformity score of a frame is the distance of the mask's box-head embedding
to the target's own pre-gap frames (centroid: 1 - cosine to the mean embedding; nn: smallest
1 - cosine to any pre-gap frame).  The pool is every valid pre-gap frame; the tested stream is
every k-th frame from the gap start on which SAM 2 claims an object (mask non-empty).  Alarm:
wealth >= 1/alpha.

Scored against the ground truth of P0:
  detected      a switched episode with an alarm in [switch_frame, switch_frame + d], d in {10, 30, 60}
  false alarm   a recovered episode with an alarm anywhere, or a switched one with an alarm
                before its switch frame
  lost episodes are reported separately (an alarm there is neither counted for nor against)
Comparators on the same episodes: SAM 2's object score (alarm when it drops below tau after the
gap start) and a one-sided CUSUM with a null learned from the pool.  All three are swept into a
ROC: x = false-alarm rate on recovered episodes, y = detection rate within 30 frames.
Writes out/p1_<run>.{json,txt} and data/runs/p1_<run>/<episode>.npz (wealth traces).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget.martingale import SwitchDetector, cusum_learned_null  # noqa: E402

try:
    from certtrack.harness import write_out
except Exception:  # pragma: no cover
    def write_out(stem, payload, text):
        (ROOT / "out").mkdir(exist_ok=True)
        (ROOT / "out" / f"{stem}.json").write_text(json.dumps(payload, indent=1, default=str))
        (ROOT / "out" / f"{stem}.txt").write_text(text)
        return ROOT / "out" / f"{stem}.json", ROOT / "out" / f"{stem}.txt"

ALPHAS = (0.2, 0.1, 0.05, 0.02, 0.01)
WINDOWS = (10, 30, 60)


def nonconformity(embs, valid, pre_idx, kind):
    """Score per frame (nan where invalid); the pool is the valid pre-gap frames."""
    pool = embs[pre_idx]
    if kind == "centroid":
        c = pool.mean(0)
        c /= max(np.linalg.norm(c), 1e-9)
        s = 1.0 - embs @ c
    else:
        d = 1.0 - embs @ pool.T                        # (T, n_pool)
        s = np.where(np.eye(embs.shape[0], dtype=bool)[:, :0].any(1), 0, d.min(1)) if False else d.min(1)
        # a pre-gap frame scored against the pool would see itself: leave-one-out for those
        for j, i in enumerate(pre_idx):
            dj = np.delete(d[i], j)
            s[i] = dj.min() if dj.size else 0.0
    s = np.where(valid, s, np.nan)
    return s


def episode_alarms(rec, z, kind, k, seed=0):
    frames = z["frames"]
    embs = z["embs"].astype(np.float64)
    valid = (z["boxes"][:, 0] == z["boxes"][:, 0]) & (np.linalg.norm(embs, axis=1) > 0.5)
    gap_start, gap_end = rec["gap_start"], rec["gap_end"]
    pre = np.nonzero((frames < gap_start) & valid)[0]
    if pre.size < 5:
        return None
    s = nonconformity(embs, valid, pre, kind)
    pool = s[pre]
    tested = [i for i in range(len(frames)) if frames[i] >= gap_start and valid[i] and (frames[i] - gap_start) % k == 0]
    out = {"frames_tested": [int(frames[i]) for i in tested], "pool_n": int(pre.size)}
    # martingales
    for mk in ("mixture", "jumper"):
        det = SwitchDetector(pool, kind=mk, alpha=1.0, seed=seed)
        w = [det.update(s[i])[1] for i in tested]
        out[f"wealth_{mk}"] = w
    # CUSUM on the same tested scores with the pool as the null
    series = np.concatenate([pool, [s[i] for i in tested]])
    out["cusum"] = cusum_learned_null(series, n_cal=pool.size).tolist()
    # SAM 2's object score after the gap start (every frame)
    post = [i for i in range(len(frames)) if frames[i] >= gap_start]
    out["frames_score"] = [int(frames[i]) for i in post]
    out["obj_score"] = [float(z["scores"][i]) for i in post]
    return out


def first_frame(frames, values, thr, above=True):
    for f, v in zip(frames, values):
        if (v >= thr) if above else (v < thr):
            return int(f)
    return None


def evaluate(records, alarms, detector, thr, d=30):
    """(detected, n_switched, false, n_recovered, n_lost_alarmed, n_lost, delays)."""
    det = fa = n_sw = n_rec = n_lost = lost_al = 0
    delays = []
    for rec, al in zip(records, alarms):
        if al is None:
            continue
        if detector in ("mixture", "jumper"):
            t_al = first_frame(al["frames_tested"], al[f"wealth_{detector}"], 1.0 / thr)
        elif detector == "cusum":
            t_al = first_frame(al["frames_tested"], al["cusum"], thr)
        else:
            t_al = first_frame(al["frames_score"], al["obj_score"], thr, above=False)
        if rec["outcome"] == "switched":
            n_sw += 1
            sf = rec["switch_frame"]
            if t_al is not None and t_al < sf:
                fa += 1
            elif t_al is not None and t_al <= sf + d:
                det += 1
                delays.append(t_al - sf)
        elif rec["outcome"] == "recovered":
            n_rec += 1
            fa += t_al is not None
        else:
            n_lost += 1
            lost_al += t_al is not None
    return {"detected": det, "n_switched": n_sw, "false": fa, "n_recovered": n_rec, "lost_alarmed": lost_al,
            "n_lost": n_lost, "median_delay": float(np.median(delays)) if delays else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p0_sam2_bp_dev")
    ap.add_argument("--k", nargs="*", type=int, default=[1, 4, 8, 12])
    ap.add_argument("--score", nargs="*", default=["centroid", "nn"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    payload = json.loads((ROOT / "out" / f"{a.run}.json").read_text())
    records = payload["records"]
    runs = ROOT / "data" / "runs" / a.run
    out_dir = ROOT / "data" / "runs" / f"p1_{a.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = [np.load(runs / f"{r['seq']}_{r['id']}_{r['gap_start']}.npz") for r in records]
    results = {}
    lines = [f"P1: conformal switch detection on {a.run}, {time.strftime('%Y-%m-%d')}", "",
             f"episodes: {len(records)}  switched {sum(r['outcome'] == 'switched' for r in records)}  "
             f"recovered {sum(r['outcome'] == 'recovered' for r in records)}  lost {sum(r['outcome'] == 'lost' for r in records)}", ""]
    t0 = time.time()
    for kind in a.score:
        for k in a.k:
            alarms = [episode_alarms(r, z, kind, k, a.seed) for r, z in zip(records, zs)]
            key = f"{kind}_k{k}"
            results[key] = {"per_episode": alarms}
            lines.append(f"score {kind}, every {k} frame(s)")
            lines.append(f"  {'detector':9s} {'thr':>6s} | {'det/sw':>7s} {'FA/rec':>7s} {'delay':>6s} | {'det10':>5s} {'det60':>5s} {'lost+al':>7s}")
            table = {}
            for detc, thrs in (("mixture", ALPHAS), ("jumper", ALPHAS), ("cusum", (2, 4, 6, 8, 12, 16)),
                               ("objscore", (-2, 0, 2, 4, 6, 8))):
                table[detc] = []
                for thr in thrs:
                    e30 = evaluate(records, alarms, detc, thr, 30)
                    e10 = evaluate(records, alarms, detc, thr, 10)
                    e60 = evaluate(records, alarms, detc, thr, 60)
                    row = {"thr": thr, **e30, "detected_10": e10["detected"], "detected_60": e60["detected"]}
                    table[detc].append(row)
                    lines.append(f"  {detc:9s} {thr:6g} | {e30['detected']:3d}/{e30['n_switched']:<3d} {e30['false']:3d}/{e30['n_recovered']:<3d} "
                                 f"{e30['median_delay']:6.1f} | {e10['detected']:5d} {e60['detected']:5d} {e30['lost_alarmed']:3d}/{e30['n_lost']:<3d}")
            results[key]["table"] = table
            lines.append("")
            for r, al in zip(records, alarms):
                if al is not None:
                    np.savez_compressed(out_dir / f"{r['seq']}_{r['id']}_{r['gap_start']}_{key}.npz",
                                        **{kk: np.asarray(v) for kk, v in al.items()})
    lines.append("det/sw: switched episodes with an alarm within 30 frames of the switch, over switched; FA/rec: recovered episodes "
                 "with any alarm (plus switched ones alarmed before the switch), over recovered; delay: median frames from switch to alarm; "
                 "thr is alpha for the martingales (alarm at wealth 1/alpha), h for the CUSUM, the object-score level for objscore.")
    text = "\n".join(lines) + "\n"
    pj, pt = write_out(f"p1_{a.run}", {"run": a.run, "args": vars(a), "results": {k: v["table"] for k, v in results.items()},
                                        "seconds": time.time() - t0}, text)
    print(text)
    print(f"wrote {pj}")


if __name__ == "__main__":
    main()
