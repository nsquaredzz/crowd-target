"""SAM 2.1 as a per-frame stepping tracker with its memory bank exposed.

Facts this relies on (Meta's sam2, pinned commit in sam2_commit.txt; re-check there):
- `init_state` builds `state["images"]` from a JPEG folder; afterwards the predictor only does
  `state["images"][i]` and `len(...)`, so the tensor is replaced by a LazyFrames object.
- `propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0)` processes frame t
  alone and yields (t, obj_ids, video_res_masks); this is certified-skip's gate_sam2.py pattern.
- The per-frame object score is not yielded; it sits in
  `state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"][t]["object_score_logits"]`
  (or under "cond_frame_outputs" for prompted frames); <= 0 means SAM 2 judged the object absent
  and set the mask logits to -1024.
- Memory for frame t+1 is read by exact frame index from those dicts (the 6 most recent
  non-conditioning frames, all conditioning frames, up to 15 object pointers), so popping an
  entry removes that frame from memory and moving an entry to "cond_frame_outputs" makes it a
  permanent exemplar.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"t": "configs/sam2.1/sam2.1_hiera_t.yaml", "s": "configs/sam2.1/sam2.1_hiera_s.yaml",
           "b+": "configs/sam2.1/sam2.1_hiera_b+.yaml", "l": "configs/sam2.1/sam2.1_hiera_l.yaml"}
CKPT_NAMES = {"t": "sam2.1_hiera_tiny.pt", "s": "sam2.1_hiera_small.pt", "b+": "sam2.1_hiera_base_plus.pt",
              "l": "sam2.1_hiera_large.pt"}
CKPT_DIRS = [ROOT / "data" / "sam2", ROOT.parent / "certified-skip" / "data" / "sam2"]


def pick_device(pref: str | None = None) -> str:
    import torch
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_checkpoint(model: str) -> Path:
    for d in CKPT_DIRS:
        p = d / CKPT_NAMES[model]
        if p.exists():
            return p
    raise FileNotFoundError(f"no {CKPT_NAMES[model]} under {[str(d) for d in CKPT_DIRS]}")


def build(model: str = "b+", device: str | None = None, overrides: list[str] = ()):
    from sam2.build_sam import build_sam2_video_predictor
    dev = pick_device(device)
    pred = build_sam2_video_predictor(CONFIGS[model], str(find_checkpoint(model)), device=dev,
                                      hydra_overrides_extra=list(overrides))
    pred.crowdtarget_device = dev
    pred.crowdtarget_model = model
    return pred


def make_state(pred, lazy) -> dict:
    """An inference state over a LazyFrames object (no JPEG copies of the window)."""
    import torch
    with tempfile.TemporaryDirectory() as d:
        Image.new("RGB", (lazy.video_width, lazy.video_height)).save(Path(d) / "00000.jpg")
        state = pred.init_state(video_path=d, offload_video_to_cpu=True, offload_state_to_cpu=False)
    state["images"] = lazy
    state["num_frames"] = len(lazy)
    state["video_height"] = lazy.video_height
    state["video_width"] = lazy.video_width
    state["cached_features"] = {}
    state["frames_tracked_per_obj"] = {} if "frames_tracked_per_obj" in state else state.get("frames_tracked_per_obj", {})
    torch.mps.empty_cache() if pred.crowdtarget_device == "mps" else None
    return state


def prompt_box(pred, state, t: int, obj_id: int, box_xyxy) -> None:
    pred.add_new_points_or_box(state, frame_idx=int(t), obj_id=int(obj_id),
                               box=np.asarray(box_xyxy, np.float32))


def prompt_points(pred, state, t: int, obj_id: int, points_xy, labels) -> None:
    """Positive (1) and negative (0) clicks on frame t, in video pixel coordinates."""
    pred.add_new_points_or_box(state, frame_idx=int(t), obj_id=int(obj_id),
                               points=np.asarray(points_xy, np.float32).reshape(-1, 2), labels=np.asarray(labels, np.int32).reshape(-1))


def prompt_mask(pred, state, t: int, obj_id: int, mask: np.ndarray) -> None:
    pred.add_new_mask(state, frame_idx=int(t), obj_id=int(obj_id), mask=np.asarray(mask, bool))


def _obj_idx(state, obj_id: int) -> int:
    return state["obj_id_to_idx"][obj_id]


def object_score(state, obj_id: int, t: int) -> float:
    d = state["output_dict_per_obj"][_obj_idx(state, obj_id)]
    for key in ("non_cond_frame_outputs", "cond_frame_outputs"):
        if t in d[key]:
            return float(d[key][t]["object_score_logits"].reshape(-1)[0])
    return float("nan")


def step(pred, state, t: int) -> dict[int, tuple[np.ndarray, float]]:
    """Run SAM 2 on frame t alone; {obj_id: (mask (H, W) bool, object score)}."""
    import torch
    out = {}
    with torch.inference_mode():
        for fi, obj_ids, logits in pred.propagate_in_video(state, start_frame_idx=int(t), max_frame_num_to_track=0):
            m = (logits[:, 0] > 0).cpu().numpy()
            for j, oid in enumerate(obj_ids):
                out[int(oid)] = (m[j], object_score(state, int(oid), int(fi)))
    return out


def pop_noncond(state, obj_id: int, t: int):
    """Remove frame t from the object's memory (used while the target is judged absent)."""
    return state["output_dict_per_obj"][_obj_idx(state, obj_id)]["non_cond_frame_outputs"].pop(t, None)


def promote_to_cond(state, obj_id: int, t: int) -> bool:
    """Make frame t a permanent exemplar in the object's memory."""
    d = state["output_dict_per_obj"][_obj_idx(state, obj_id)]
    if t in d["non_cond_frame_outputs"]:
        d["cond_frame_outputs"][t] = d["non_cond_frame_outputs"].pop(t)
        return True
    return False


def memory_keys(state, obj_id: int) -> tuple[list[int], list[int]]:
    d = state["output_dict_per_obj"][_obj_idx(state, obj_id)]
    return sorted(d["cond_frame_outputs"]), sorted(d["non_cond_frame_outputs"])


def timed_step(pred, state, t: int):
    import torch
    t0 = time.perf_counter()
    out = step(pred, state, t)
    if pred.crowdtarget_device == "mps":
        torch.mps.synchronize()
    return out, time.perf_counter() - t0
