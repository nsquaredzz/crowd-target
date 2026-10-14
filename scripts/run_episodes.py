#!/usr/bin/env python3
"""P0: SAM 2.1 alone on the single-target episodes, logging what Piece 1 needs.

    python scripts/run_episodes.py --set dev --model b+
    python scripts/run_episodes.py --set dev --limit 3        # smoke

Per episode: prompt SAM 2 with the ground-truth box at t_start, step every frame to t_end with
the memory as shipped, log per frame the mask (stored at 1/4 resolution, bit-packed), the object
score, the mask's box, the box-head embedding of the mask's box, and the outcome against the
ground truth.  Writes data/runs/<stem>/<seq>_<id>_<gap_start>.npz and out/<stem>.{json,txt}.
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

from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import fixes  # noqa: E402
from crowdtarget import metrics as M  # noqa: E402
from crowdtarget import sam2io  # noqa: E402
from crowdtarget.data.frames import FrameSource, LazyFrames  # noqa: E402
from crowdtarget.data.mot import dets_by_frame, frame_path, read_det, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.embed import BoxHeadEmbedder, mask_bbox  # noqa: E402
from crowdtarget.harness import machine_info as _machine_info, write_out  # noqa: E402


def machine_info(pred):
    return _machine_info({"device": pred.crowdtarget_device, "model": pred.crowdtarget_model})


def sound_ctx(ep, gt, seq_dir, src, det_mode: str, imgsz: int = 1920) -> dict:
    """What the sound tracker needs besides SAM 2: the background model, detections, the speed bound, frames."""
    from census import speed_bound
    import detect
    bg = np.load(ROOT / "data" / "cache" / f"{ep.seq}_bg.npy")
    if det_mode == "yolo":
        dets = detect.ensure(ep.seq, range(ep.t_start, ep.t_end + 1), imgsz)
    else:
        dbf = dets_by_frame(read_det(seq_dir))
        dets = {t: (b.astype(np.float32), np.ones(len(b), np.float32)) for t, b in dbf.items()}
    return {"bg": bg, "dets": dets, "v_max": speed_bound(gt), "frames": src, "t_start": ep.t_start}


def box_iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_episode(ep: E.Episode, pred, embedder, gt, info, seq_dir, down: int = 4, fix_kind: str | None = None, det_mode: str = "yolo"):
    paths = [frame_path(seq_dir, t, info) for t in range(ep.t_start, ep.t_end + 1)]
    src = FrameSource(paths)
    lazy = LazyFrames(src)
    state = sam2io.make_state(pred, lazy)
    sam2io.prompt_box(pred, state, 0, 1, ep.start_box)
    T = len(src)
    H, W = src.H, src.W
    ctx = sound_ctx(ep, gt, seq_dir, src, det_mode) if fix_kind in ("sound", "sound2", "sliver") else {}
    fix = fixes.make_fix(fix_kind, pred, W, H, ep.start_box, **ctx)
    if fix:
        fix.install()
    every = fix_kind in ("sound", "sound2", "sliver")
    diag = []
    masks_small = np.zeros((T, H // down, W // down), bool)
    scores = np.full(T, np.nan, np.float32)
    boxes = np.full((T, 4), np.nan, np.float32)
    embs = np.zeros((T, 1024), np.float32)
    outcomes = []
    secs = []
    tg = targets(gt)
    for i in range(T):
        t = ep.t_start + i
        if fix and (i > 0 or every):
            fix.before_step(state, i)
        out, dt = sam2io.timed_step(pred, state, i)
        secs.append(dt)
        mask, score = out[1]
        if fix and (i > 0 or every):
            diag.append(fix.after_step(state, i, mask, score))
            if hasattr(fix, "out_mask"):
                mask = fix.out_mask
        scores[i] = score
        masks_small[i] = mask[::down, ::down][: H // down, : W // down]
        bb = mask_bbox(mask)
        if bb is not None:
            boxes[i] = bb
            embs[i] = embedder(src[i], bb[None], key=i)[0]
        outcomes.append(M.frame_outcome(mask, tg, t, ep.id))
        src._cache.pop(i - 2, None)
        lazy._cache.pop(i - 2, None)
    vis_post = {int(r[0]): bool(r[8] >= 0.5) for r in tg[(tg[:, 1] == ep.id) & (tg[:, 0] > ep.gap_end)]}
    res = M.episode_outcome(outcomes, list(range(ep.t_start, ep.t_end + 1)), ep.gap_start, ep.gap_end, vis_post)
    if every and diag:
        # ambiguity: post frames where the tracker held a set that contained the true person
        tg_rows = {int(r[0]): r for r in tg[tg[:, 1] == ep.id]}
        amb_true = amb = 0
        for i, dg in enumerate(diag):
            t = ep.t_start + i
            if t <= ep.gap_end or not vis_post.get(t, False) or dg.get("state_after") != "ambiguous":
                continue
            amb += 1
            r = tg_rows.get(t)
            if r is not None and any(box_iou_xyxy(c, [r[2], r[3], r[2] + r[4], r[3] + r[5]]) >= 0.5 for c in dg.get("cands", [])):
                amb_true += 1
        n_post = sum(1 for t, v in vis_post.items() if v and t <= ep.t_end)
        res["amb_post_frac"] = amb / max(n_post, 1)
        res["amb_true_frac"] = amb_true / max(n_post, 1)
        res["commits"] = [(ep.t_start + i, dg["commit"]) for i, dg in enumerate(diag) if dg.get("commit")]
        # the firm-only view: provisional frames (a commit on probation) count as absent
        firm_outcomes = [o if dg.get("firm_after") else "absent" for o, dg in zip(outcomes, diag)]
        rf = M.episode_outcome(firm_outcomes, list(range(ep.t_start, ep.t_end + 1)), ep.gap_start, ep.gap_end, vis_post)
        res["firm"] = {k: rf[k] for k in ("outcome", "switch_id", "switch_frame", "switch_phase", "post_correct_frac", "delay")}
        res["vetoes"] = int(sum(1 for dg in diag if dg.get("veto")))
        res["absent_frames"] = int(sum(1 for dg in diag if not dg.get("accepted")))
        if res["outcome"] == "lost" and res["amb_true_frac"] >= 0.5:
            res["outcome"] = "ambiguous"
    res.update({"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "gap_end": ep.gap_end, "g": ep.g, "bin": ep.bin,
                "t_start": ep.t_start, "t_end": ep.t_end, "s_per_frame": float(np.mean(secs)),
                "first_absent_frame": next((ep.t_start + i for i in range(T) if scores[i] <= 0), None)})
    arrays = {"masks_small": np.packbits(masks_small, axis=-1), "mask_shape": np.array(masks_small.shape), "scores": scores,
              "boxes": boxes, "embs": embs, "outcomes": np.array(outcomes), "frames": np.arange(ep.t_start, ep.t_end + 1),
              "secs": np.array(secs), "diag": np.array(json.dumps(diag))}
    if fix:
        fix.uninstall()
    del state
    return res, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "held", "all"])
    ap.add_argument("--model", default="b+", choices=["t", "s", "b+", "l"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fix", default=None, choices=[None, "samurai", "dam", "sound", "sound2", "sliver"], help="fix to apply (crowdtarget/fixes.py, controller.py)")
    ap.add_argument("--det", default="yolo", choices=["yolo", "public"], help="detections for the sound tracker's bookkeeping")
    ap.add_argument("--first", default=None, help="comma list of <seq>_<id> to run before the rest")
    a = ap.parse_args()
    eps = E.read(ROOT / "data" / "episodes" / f"mot20_{a.set}.json")
    if a.first:
        first = a.first.split(",")
        eps = sorted(eps, key=lambda e: (f"{e.seq}_{e.id}" not in first, first.index(f"{e.seq}_{e.id}") if f"{e.seq}_{e.id}" in first else 0))
    if a.limit:
        eps = eps[: a.limit]
    stem = a.out or ((f"fix_{a.fix}_{a.det}_{a.set}" if a.fix in ("sound", "sound2", "sliver") else f"fix_{a.fix}_{a.set}") if a.fix else f"p0_sam2_{a.model.replace('+', 'p')}_{a.set}")
    runs = ROOT / "data" / "runs" / stem
    runs.mkdir(parents=True, exist_ok=True)
    pred = sam2io.build(a.model, a.device)
    embedder = BoxHeadEmbedder(pred.crowdtarget_device)
    info_cache, gt_cache = {}, {}
    records = []
    prev = ROOT / "out" / f"{stem}.json"
    old = {(r["seq"], r["id"], r["gap_start"]): r for r in json.loads(prev.read_text())["records"]} if prev.exists() else {}
    t_all = time.time()
    for k, ep in enumerate(eps):
        seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
        if ep.seq not in gt_cache:
            gt_cache[ep.seq] = read_gt(seq_dir)
            info_cache[ep.seq] = seq_info(seq_dir)
        key = (ep.seq, ep.id, ep.gap_start)
        if key in old and (runs / f"{ep.seq}_{ep.id}_{ep.gap_start}.npz").exists():
            records.append(old[key])      # resume: keep what an earlier run finished
            print(f"[{k + 1}/{len(eps)}] {ep.seq} id {ep.id}: kept from the earlier run ({old[key]['outcome']})", flush=True)
            continue
        t0 = time.time()
        res, arrays = run_episode(ep, pred, embedder, gt_cache[ep.seq], info_cache[ep.seq], seq_dir, fix_kind=a.fix, det_mode=a.det)
        np.savez_compressed(runs / f"{ep.seq}_{ep.id}_{ep.gap_start}.npz", **arrays)
        res["seconds"] = round(time.time() - t0, 1)
        records.append(res)
        print(f"[{k + 1}/{len(eps)}] {ep.seq} id {ep.id} g {ep.g:3d} {ep.bin:>6s}: {res['outcome']:9s} "
              f"{('-> ' + str(res['switch_id']) + ' at ' + str(res['switch_frame']) + ' (' + str(res['switch_phase']) + ')') if res['outcome'] == 'switched' else ''} "
              f"post-correct {100 * res['post_correct_frac']:.0f} %  "
              f"{('firm-only ' + res['firm']['outcome'] + ' ' + format(100 * res['firm']['post_correct_frac'], '.0f') + ' %  amb ' + format(100 * res['amb_post_frac'], '.0f') + ' % (true in set ' + format(100 * res['amb_true_frac'], '.0f') + ' %) commits ' + str(len(res['commits'])) + ' vetoes ' + str(res['vetoes']) + '  ') if 'amb_post_frac' in res else ''}"
              f"{res['s_per_frame']:.2f} s/frame  {res['seconds']} s", flush=True)
        payload = {"machine": machine_info(pred), "set": a.set, "fix": a.fix, "records": records, "summary": M.summarise(records),
                   "seconds": round(time.time() - t_all, 1)}
        write_out(stem, payload, M.format_summary(payload["summary"], f"SAM 2.1 {a.model} {('+ ' + a.fix) if a.fix else 'alone'} on MOT20 episodes ({a.set}), {time.strftime('%Y-%m-%d')}"))
    print(M.format_summary(M.summarise(records), stem))


if __name__ == "__main__":
    main()
