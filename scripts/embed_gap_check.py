#!/usr/bin/env python3
"""Separability ACROSS an occlusion gap, on the episodes themselves, from ground-truth crops.

    python scripts/embed_gap_check.py --set dev

For each episode: the target's crops at up to 8 pre-gap frames (visibility >= 0.5) form the pool;
its crops at up to 8 post-gap frames (visibility >= 0.5) are the same-person-across-gap sample;
every other identity's crop (visibility >= 0.5, height >= 40) at those post frames is the
other-person sample.  Reported: AUC(same-across-gap closer than other), and the share of
same-across-gap scores above the pool's 90th and 100th percentile: that is what a conformal
p-value on the recovered target sees after the gap, i.e. the false-alarm pressure that no
martingale can remove.  Writes out/embed_gap_check_<set>.{json,txt}.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget import episodes as E  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.embed import make_embedder  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1
    rp = ranks[: pos.size].sum()
    return float(1.0 - (rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def crops_at(gt, t, ids=None, min_vis=0.5, min_h=40):
    r = gt[(gt[:, 0] == t) & (gt[:, 8] >= min_vis) & (gt[:, 5] >= min_h)]
    if ids is not None:
        r = r[np.isin(r[:, 1], ids)]
    boxes = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], -1) if r.shape[0] else np.zeros((0, 4))
    return r[:, 1].astype(int), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pool-span", type=int, default=0,
                    help="pool from this many frames before the gap instead of the episode's 30-frame pre window (0 = pre window)")
    ap.add_argument("--embedding", default="frcnn", choices=["frcnn", "dino"])
    a = ap.parse_args()
    eps = E.read(ROOT / "data" / "episodes" / f"mot20_{a.set}.json")[: a.limit]
    emb = make_embedder(a.embedding)
    t0 = time.time()
    same_all, other_all, per = [], [], []
    cache = {}
    for ep in eps:
        seq = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
        if ep.seq not in cache:
            cache[ep.seq] = (seq_info(seq), targets(read_gt(seq)))
        info, gt = cache[ep.seq]
        tr = gt[gt[:, 1] == ep.id]
        t_lo = ep.t_start if not a.pool_span else max(1, ep.gap_start - a.pool_span)
        pre_t = tr[(tr[:, 0] >= t_lo) & (tr[:, 0] < ep.gap_start) & (tr[:, 8] >= 0.5)][:, 0].astype(int)
        post_t = tr[(tr[:, 0] > ep.gap_end) & (tr[:, 0] <= ep.t_end) & (tr[:, 8] >= 0.5)][:, 0].astype(int)
        pre_t = pre_t[np.linspace(0, len(pre_t) - 1, min(a.n, len(pre_t))).astype(int)] if len(pre_t) else pre_t
        post_t = post_t[np.linspace(0, len(post_t) - 1, min(a.n, len(post_t))).astype(int)] if len(post_t) else post_t
        pool, same, other = [], [], []
        for t in pre_t:
            ids, boxes = crops_at(gt, t, [ep.id])
            if boxes.shape[0]:
                pool.append(emb(np.asarray(Image.open(frame_path(seq, int(t), info)).convert("RGB")), boxes)[0])
        if len(pool) < 3:
            continue
        pool = np.stack(pool)
        c = pool.mean(0)
        c /= np.linalg.norm(c)
        pool_s = 1 - pool @ c
        for t in post_t:
            ids, boxes = crops_at(gt, t)
            if boxes.shape[0] == 0:
                continue
            e = emb(np.asarray(Image.open(frame_path(seq, int(t), info)).convert("RGB")), boxes)
            s = 1 - e @ c
            for i, id_ in enumerate(ids):
                (same if id_ == ep.id else other).append(float(s[i]))
        if not same or not other:
            continue
        same, other = np.asarray(same), np.asarray(other)
        d = {"episode": f"{ep.seq}_{ep.id}_{ep.gap_start}", "g": ep.g, "bin": ep.bin, "pool_n": int(pool.shape[0]),
             "pool_p90": float(np.quantile(pool_s, 0.9)), "pool_max": float(pool_s.max()),
             "same_median": float(np.median(same)), "other_median": float(np.median(other)), "other_min": float(other.min()),
             "same_above_pool_p90": float(np.mean(same > np.quantile(pool_s, 0.9))), "same_above_pool_max": float(np.mean(same > pool_s.max())),
             "auc": auc(same, other), "n_same": int(same.size), "n_other": int(other.size)}
        per.append(d)
        same_all += list(same)
        other_all += list(other)
        print(f"{d['episode']:22s} g {ep.g:3d} pool p90 {d['pool_p90']:.3f} max {d['pool_max']:.3f} | same-across-gap median {d['same_median']:.3f} "
              f"(above pool max {100 * d['same_above_pool_max']:.0f} %) | other median {d['other_median']:.3f} min {d['other_min']:.3f} | AUC {d['auc']:.2f}", flush=True)
    same_all, other_all = np.asarray(same_all), np.asarray(other_all)
    res = {"per_episode": per, "auc_pooled": auc(same_all, other_all) if same_all.size and other_all.size else float("nan"),
           "auc_median": float(np.median([d["auc"] for d in per])) if per else float("nan"),
           "same_above_pool_max_mean": float(np.mean([d["same_above_pool_max"] for d in per])) if per else float("nan"),
           "machine": machine_info({"device": emb.device}), "seconds": round(time.time() - t0, 1)}
    lines = [f"Across-gap separability of {emb.name} on MOT20 {a.set} episodes, {time.strftime('%Y-%m-%d')}", "",
             f"{'episode':22s} {'g':>4s} {'pool p90':>8s} {'pool max':>8s} {'same med':>8s} {'>poolmax':>8s} {'other med':>9s} {'other min':>9s} {'AUC':>5s}"]
    for d in per:
        lines.append(f"{d['episode']:22s} {d['g']:4d} {d['pool_p90']:8.3f} {d['pool_max']:8.3f} {d['same_median']:8.3f} {100 * d['same_above_pool_max']:7.0f}% "
                     f"{d['other_median']:9.3f} {d['other_min']:9.3f} {d['auc']:5.2f}")
    lines += ["", f"pooled AUC {res['auc_pooled']:.3f}  median per-episode AUC {res['auc_median']:.3f}  "
              f"mean share of same-person post-gap crops scoring above the whole pre-gap pool: {100 * res['same_above_pool_max_mean']:.0f} %",
              "A same-person crop above the whole pool gets conformal p-value ~ 1/(n+1) whatever the identity: that share is the",
              "false-alarm pressure of the embedding itself after the gap, before any martingale."]
    pj, pt = write_out(f"embed_gap_check_{a.set}" + (f"_span{a.pool_span}" if a.pool_span else "") + (f"_{a.embedding}" if a.embedding != "frcnn" else ""),
                       res, "\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print(f"wrote {pj}")


if __name__ == "__main__":
    main()
