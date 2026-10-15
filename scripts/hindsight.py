#!/usr/bin/env python3
"""Hindsight tracking on the dev episodes (protocol and kill criteria: python/crowdtarget/hindsight.py).

    python scripts/hindsight.py --forward fix_sound_yolo_dev
    python scripts/hindsight.py --forward fix_sound_yolo_dev --first MOT20-05_611 --limit 1 --out smoke_hindsight

Reads the forward run's stored masks and log, runs one reverse SAM 2.1 pass per triggered episode with
every re-emergence candidate as its own object, decides by where the backward tracks meet the forward
track, and scores the combined output against the ground truth with the same metrics as every other run.
Writes out/<out>.{json,txt} and data/runs/<out>/<episode>.npz (output masks at 1/4, candidates' meet scores).
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

from census import speed_bound  # noqa: E402
import detect  # noqa: E402
from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import hindsight as HS  # noqa: E402
from crowdtarget import metrics as M  # noqa: E402
from crowdtarget import sam2io  # noqa: E402
from crowdtarget.data.frames import FrameSource, LazyFrames  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402

DOWN = 4
W_MEET = 10
K_MAX = 16


def unpack(z):
    return np.unpackbits(z["masks_small"], axis=-1)[:, :, : z["mask_shape"][2]].astype(bool)


def bbox(m):
    ys, xs = np.nonzero(m)
    return None if ys.size == 0 else np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def gt_id_of_box(box, rows):
    if rows.shape[0] == 0:
        return -1
    G = np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1)
    x1 = np.maximum(G[:, 0], box[0]); y1 = np.maximum(G[:, 1], box[1])
    x2 = np.minimum(G[:, 2], box[2]); y2 = np.minimum(G[:, 3], box[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    u = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    io = inter / np.maximum(u, 1e-9)
    j = int(np.argmax(io))
    return int(rows[j, 1]) if io[j] >= 0.5 else -1


def backward_pass(pred, seq_dir, info, t0, t_b, boxes):
    """Reverse SAM 2.1 over frames t0..t_b with every box prompted at t_b; masks at 1/4 per object per frame."""
    paths = [frame_path(seq_dir, t, info) for t in range(t0, t_b + 1)]
    src = FrameSource(paths)
    lazy = LazyFrames(src)
    state = sam2io.make_state(pred, lazy)
    T = len(src)
    for k, b in enumerate(boxes):
        sam2io.prompt_box(pred, state, T - 1, k + 1, b)
    h4, w4 = src.H // DOWN, src.W // DOWN
    K = len(boxes)
    masks = np.zeros((K, T, h4, w4), bool)
    scores = np.full((K, T), np.nan, np.float32)
    import torch
    t_start = time.time()
    with torch.inference_mode():
        for i in range(T - 1, 0, -1):            # SAM 2 cannot process index 0 in reverse; the window starts one frame early
            for fi, obj_ids, logits in pred.propagate_in_video(state, start_frame_idx=i, max_frame_num_to_track=0, reverse=True):
                m = (logits[:, 0] > 0).cpu().numpy()
                for j, oid in enumerate(obj_ids):
                    masks[oid - 1, fi] = m[j][::DOWN, ::DOWN][:h4, :w4]
                    scores[oid - 1, fi] = sam2io.object_score(state, int(oid), int(fi))
            for d in state["output_dict_per_obj"].values():          # keep only the memory SAM 2 reads (i+1 .. i+7)
                for key in [k for k in d["non_cond_frame_outputs"] if k > i + 8]:
                    d["non_cond_frame_outputs"].pop(key)
            src._cache.pop(i + 2, None)
            lazy._cache.pop(i + 2, None)
    secs = time.time() - t_start
    del state
    if pred.crowdtarget_device == "mps":
        torch.mps.empty_cache()
    return masks, scores, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", default="fix_sound_yolo_dev")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--first", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="deploy", choices=["deploy", "oracle"],
                    help="oracle: decision 15 frames after the ground-truth re-emergence, candidates = the target and the 7 people nearest it (ground-truth boxes)")
    a = ap.parse_args()
    stem = a.out or (f"hindsight_{a.mode}_{a.forward}")
    fwd = {(r["seq"], r["id"], r["gap_start"]): r for r in json.loads((ROOT / "out" / f"{a.forward}.json").read_text())["records"]}
    eps = [e for e in E.read(ROOT / "data" / "episodes" / f"mot20_{a.set}.json") if (e.seq, e.id, e.gap_start) in fwd]
    if a.first:
        first = a.first.split(",")
        eps = sorted(eps, key=lambda e: (f"{e.seq}_{e.id}" not in first, first.index(f"{e.seq}_{e.id}") if f"{e.seq}_{e.id}" in first else 0))
    if a.limit:
        eps = eps[: a.limit]
    runs = ROOT / "data" / "runs" / stem
    runs.mkdir(parents=True, exist_ok=True)
    pred = sam2io.build("b+")
    cache = {}
    records = []
    t_all = time.time()
    for n, ep in enumerate(eps):
        key = f"{ep.seq}_{ep.id}_{ep.gap_start}"
        seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
        if ep.seq not in cache:
            gt = targets(read_gt(seq_dir))
            cache[ep.seq] = (gt, seq_info(seq_dir), speed_bound(gt))
        gt, info, vmax = cache[ep.seq]
        z = np.load(ROOT / "data" / "runs" / a.forward / f"{key}.npz")
        diag = json.loads(str(z["diag"]))
        fmask = unpack(z)
        frames = z["frames"]
        T = len(frames)
        firm = [bool(g.get("firm_after")) for g in diag]
        ia = HS.first_long_loss(firm)
        rec = {"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "gap_end": ep.gap_end, "g": ep.g, "bin": ep.bin,
               "forward_outcome": fwd[(ep.seq, ep.id, ep.gap_start)]["outcome"]}
        out_masks = fmask.copy()
        t0 = time.time()
        if ia is None:
            rec.update({"triggered": False, "decision": "forward"})
        else:
            t_a = int(frames[ia])
            t_b = int(frames[-1])
            if a.mode == "oracle":
                tg_ = gt[(gt[:, 1] == ep.id) & (gt[:, 0] >= ep.gap_end + 15) & (gt[:, 8] >= 0.5)]
                t_b = int(tg_[:, 0].min()) if len(tg_) else int(frames[-1])
                t_b = min(t_b, int(frames[-1]))
            hs = [bbox(fmask[i])[3] - bbox(fmask[i])[1] for i in range(max(0, ia - 10), ia + 1) if fmask[i].any()]
            h = float(np.median(hs)) * DOWN if hs else 100.0
            fb = bbox(fmask[ia])
            foot = np.array([(fb[0] + fb[2]) / 2, fb[3]]) * DOWN if fb is not None else None
            rows_b = gt[gt[:, 0] == t_b]
            reach = vmax * (t_b - t_a) * h + h / 2
            if a.mode == "oracle":
                vis = rows_b[(rows_b[:, 8] >= 0.5) & (rows_b[:, 5] >= 40)]
                vb = np.stack([vis[:, 2], vis[:, 3], vis[:, 2] + vis[:, 4], vis[:, 3] + vis[:, 5]], 1)
                tgt_rows = vis[vis[:, 1] == ep.id]
                if len(tgt_rows):
                    tc = vb[vis[:, 1] == ep.id][0]
                    cc = np.hypot((vb[:, 0] + vb[:, 2]) / 2 - (tc[0] + tc[2]) / 2, vb[:, 3] - tc[3])
                    cand = vb[np.argsort(cc)[:8]]
                else:
                    cand = np.zeros((0, 4))
            else:
                dets = detect.ensure(ep.seq, [t_b], 1920)
                b, c = dets[t_b]
                b = np.asarray(b, float)[np.asarray(c) >= 0.25]
                feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1) if len(b) else np.zeros((0, 2))
                dist = np.hypot(*(feet - foot).T) if (foot is not None and len(b)) else np.zeros(0)
                order = [j for j in np.argsort(dist) if dist[j] <= reach][:K_MAX]
                cand = b[order]
            cand_ids = [gt_id_of_box(x, rows_b) for x in cand]
            rec.update({"triggered": True, "t_a": t_a, "t_b": t_b, "reach_px": float(reach), "n_cands": len(cand),
                        "cand_gt_ids": cand_ids, "true_in_cands": ep.id in cand_ids})
            if len(cand) == 0:
                rec["decision"] = "absent"
                out_masks[ia + 1:] = False
            else:
                w0 = max(ia - W_MEET, 1)
                t_win0 = int(frames[w0 - 1])                       # one extra frame: index 0 is not processed in reverse
                bm, bs, secs = backward_pass(pred, seq_dir, info, t_win0, t_b, cand)
                off = (w0 - 1)                                     # backward local index j <-> forward index j + off
                F = fmask[w0:ia + 1]
                B = bm[:, 1:ia + 1 - off]
                hh, ww = min(F.shape[1], B.shape[2]), min(F.shape[2], B.shape[3])
                m = HS.meet_scores(F[:, :hh, :ww], B[:, :, :hh, :ww])
                dec = HS.decide(m)
                rec.update({"meet": m.round(3).tolist(), "decision": dec["kind"], "set_gt_ids": [cand_ids[k] for k in dec["set"]],
                            "backward_seconds": round(secs, 1), "backward_frames": int(bm.shape[1])})
                if ep.id in cand_ids:
                    kt = cand_ids.index(ep.id)
                    others = [m[k] for k in range(len(m)) if k != kt]
                    rec.update({"meet_true": float(m[kt]), "meet_best_other": float(max(others)) if others else 0.0, "rank_true": int((m > m[kt]).sum()) + 1})
                out_masks[ia + 1:] = False
                if dec["kind"] == "commit":
                    k = dec["index"]
                    rec["committed_gt_id"] = cand_ids[k]
                    for i in range(ia + 1, T):
                        j = i - off
                        if 0 <= j < bm.shape[1] and bs[k, j] > 0:
                            mm = bm[k, j]
                            out_masks[i, : mm.shape[0], : mm.shape[1]] = mm[: out_masks.shape[1], : out_masks.shape[2]]
                np.savez_compressed(runs / f"{key}_cands.npz", meet=m, boxes=cand, gt_ids=np.array(cand_ids), scores=bs,
                                    masks=np.packbits(bm, axis=-1), shape=np.array(bm.shape), t0=t_win0)
        outcomes = [M.frame_outcome(out_masks[i], gt, int(frames[i]), ep.id, scale=DOWN) for i in range(T)]
        tg = gt[gt[:, 1] == ep.id]
        vis_post = {int(r[0]): bool(r[8] >= 0.5) for r in tg[tg[:, 0] > ep.gap_end]}
        res = M.episode_outcome(outcomes, [int(f) for f in frames], ep.gap_start, ep.gap_end, vis_post)
        if rec.get("decision") == "ambiguous" and ep.id in rec.get("set_gt_ids", []) and res["outcome"] == "lost":
            res["outcome"] = "ambiguous"
        rec.update(res)
        rec["seconds"] = round(time.time() - t0, 1)
        records.append(rec)
        np.savez_compressed(runs / f"{key}.npz", masks_small=np.packbits(out_masks, axis=-1), mask_shape=np.array(out_masks.shape),
                            outcomes=np.array(outcomes), frames=frames, diag=np.array(json.dumps([{"state_after": "hindsight"}] * T)), scores=z["scores"])
        extra = ""
        if rec.get("triggered") and "meet" in rec:
            extra = (f"t_a {rec['t_a']} t_b {rec['t_b']} cands {rec['n_cands']} true-in {rec['true_in_cands']} meet true "
                     f"{rec.get('meet_true', float('nan')):.2f} best-other {rec.get('meet_best_other', float('nan')):.2f} -> {rec['decision']}"
                     f"{(' ' + str(rec.get('committed_gt_id'))) if rec['decision'] == 'commit' else ''}  {rec['backward_seconds']} s / {rec['backward_frames']} frames")
        print(f"[{n + 1}/{len(eps)}] {key:22s} forward {rec['forward_outcome']:9s} -> hindsight {res['outcome']:9s} "
              f"{('-> ' + str(res['switch_id'])) if res['outcome'] == 'switched' else ''} {extra}", flush=True)
        payload = {"machine": machine_info({"device": pred.crowdtarget_device}), "forward": a.forward, "records": records,
                   "summary": M.summarise(records), "seconds": round(time.time() - t_all, 1)}
        write_out(stem, payload, M.format_summary(payload["summary"], f"Hindsight on {a.forward} ({a.set}), {time.strftime('%Y-%m-%d')}"))
    print(M.format_summary(M.summarise(records), stem))


if __name__ == "__main__":
    main()
