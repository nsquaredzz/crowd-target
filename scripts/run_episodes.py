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

from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import metrics as M  # noqa: E402
from crowdtarget import sam2io  # noqa: E402
from crowdtarget.data.frames import FrameSource, LazyFrames  # noqa: E402
from crowdtarget.data.mot import frame_path, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.embed import BoxHeadEmbedder, mask_bbox  # noqa: E402

try:
    from certtrack.harness import write_out
except Exception:  # pragma: no cover
    def write_out(stem, payload, text):
        (ROOT / "out").mkdir(exist_ok=True)
        (ROOT / "out" / f"{stem}.json").write_text(json.dumps(payload, indent=1, default=str))
        (ROOT / "out" / f"{stem}.txt").write_text(text)
        return ROOT / "out" / f"{stem}.json", ROOT / "out" / f"{stem}.txt"


def machine_info(pred):
    import platform
    import torch
    return {"platform": platform.platform(), "torch": torch.__version__, "device": pred.crowdtarget_device,
            "sam2_commit": (ROOT / "sam2_commit.txt").read_text().strip() if (ROOT / "sam2_commit.txt").exists() else None,
            "model": pred.crowdtarget_model}


def run_episode(ep: E.Episode, pred, embedder, gt, info, seq_dir, down: int = 4):
    paths = [frame_path(seq_dir, t, info) for t in range(ep.t_start, ep.t_end + 1)]
    src = FrameSource(paths)
    lazy = LazyFrames(src)
    state = sam2io.make_state(pred, lazy)
    sam2io.prompt_box(pred, state, 0, 1, ep.start_box)
    T = len(src)
    H, W = src.H, src.W
    masks_small = np.zeros((T, H // down, W // down), bool)
    scores = np.full(T, np.nan, np.float32)
    boxes = np.full((T, 4), np.nan, np.float32)
    embs = np.zeros((T, 1024), np.float32)
    outcomes = []
    secs = []
    tg = targets(gt)
    for i in range(T):
        t = ep.t_start + i
        out, dt = sam2io.timed_step(pred, state, i)
        secs.append(dt)
        mask, score = out[1]
        scores[i] = score
        masks_small[i] = mask[::down, ::down]
        bb = mask_bbox(mask)
        if bb is not None:
            boxes[i] = bb
            embs[i] = embedder(src[i], bb[None], key=i)[0]
        outcomes.append(M.frame_outcome(mask, tg, t, ep.id))
        src._cache.pop(i - 2, None)
        lazy._cache.pop(i - 2, None)
    vis_post = {int(r[0]): bool(r[8] >= 0.5) for r in tg[(tg[:, 1] == ep.id) & (tg[:, 0] > ep.gap_end)]}
    res = M.episode_outcome(outcomes, list(range(ep.t_start, ep.t_end + 1)), ep.gap_start, ep.gap_end, vis_post)
    res.update({"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "gap_end": ep.gap_end, "g": ep.g, "bin": ep.bin,
                "t_start": ep.t_start, "t_end": ep.t_end, "s_per_frame": float(np.mean(secs)),
                "first_absent_frame": next((ep.t_start + i for i in range(T) if scores[i] <= 0), None)})
    arrays = {"masks_small": np.packbits(masks_small, axis=-1), "mask_shape": np.array(masks_small.shape), "scores": scores,
              "boxes": boxes, "embs": embs, "outcomes": np.array(outcomes), "frames": np.arange(ep.t_start, ep.t_end + 1),
              "secs": np.array(secs)}
    del state
    return res, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "held", "all"])
    ap.add_argument("--model", default="b+", choices=["t", "s", "b+", "l"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = E.read(ROOT / "data" / "episodes" / f"mot20_{a.set}.json")
    if a.limit:
        eps = eps[: a.limit]
    stem = a.out or f"p0_sam2_{a.model.replace('+', 'p')}_{a.set}"
    runs = ROOT / "data" / "runs" / stem
    runs.mkdir(parents=True, exist_ok=True)
    pred = sam2io.build(a.model, a.device)
    embedder = BoxHeadEmbedder(pred.crowdtarget_device)
    info_cache, gt_cache = {}, {}
    records = []
    t_all = time.time()
    for k, ep in enumerate(eps):
        seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / ep.seq
        if ep.seq not in gt_cache:
            gt_cache[ep.seq] = read_gt(seq_dir)
            info_cache[ep.seq] = seq_info(seq_dir)
        t0 = time.time()
        res, arrays = run_episode(ep, pred, embedder, gt_cache[ep.seq], info_cache[ep.seq], seq_dir)
        np.savez_compressed(runs / f"{ep.seq}_{ep.id}_{ep.gap_start}.npz", **arrays)
        res["seconds"] = round(time.time() - t0, 1)
        records.append(res)
        print(f"[{k + 1}/{len(eps)}] {ep.seq} id {ep.id} g {ep.g:3d} {ep.bin:>6s}: {res['outcome']:9s} "
              f"{('-> ' + str(res['switch_id']) + ' at ' + str(res['switch_frame']) + ' (' + str(res['switch_phase']) + ')') if res['outcome'] == 'switched' else ''} "
              f"post-correct {100 * res['post_correct_frac']:.0f} %  {res['s_per_frame']:.2f} s/frame  {res['seconds']} s", flush=True)
        payload = {"machine": machine_info(pred), "set": a.set, "records": records, "summary": M.summarise(records),
                   "seconds": round(time.time() - t_all, 1)}
        write_out(stem, payload, M.format_summary(payload["summary"], f"SAM 2.1 {a.model} alone on MOT20 episodes ({a.set}), {time.strftime('%Y-%m-%d')}"))
    print(M.format_summary(M.summarise(records), stem))


if __name__ == "__main__":
    main()
