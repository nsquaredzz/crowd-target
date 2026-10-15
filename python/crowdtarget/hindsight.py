"""Hindsight tracking: decide who came out of the occlusion by tracking every candidate backward.

What a human annotator does with a recorded clip, and the forward tracker cannot: look at who is
there after the occlusion, follow each of them backward in time, and see whose path meets the
person where they were last seen for certain.

Protocol, per episode (all of it deployment-faithful except the decision frame, see below):
  forward    the sound tracker (variant v1) runs forward; its own log gives the anchor t_a, the
             last firm frame before its first run of >= 5 non-firm frames.  If there is no such
             run, the forward output stands and hindsight is not triggered.
  decision   t_b, the end of the recorded clip.  The episode rules put it 60 frames after the
             ground-truth re-emergence; a deployed tracker would try a few horizons.  This is the
             one place the protocol leans on the episode definition, and it is stated as such.
  candidates detections (YOLO11x, conf >= 0.25) at t_b whose foot point is within the reach of the
             last firm foot point, v_max * (t_b - t_a) * h + h / 2, at most 16, nearest first.
  backward   one SAM 2.1 state over the window [t_a - W, t_b], every candidate prompted with its box
             at t_b as its own object, propagated in reverse to t_a - W.  Objects are tracked
             jointly; nothing about the target's appearance is given to the backward pass.
  meet       m_c = median over the W + 1 frames t_a - W .. t_a of IoU(backward mask of c, forward
             mask of the target).  The forward masks there are the tracker's firm output.
  decide     commit to c* = argmax m_c if m_1 >= 0.5 and m_2 < 0.25; otherwise the set
             {c : m_c >= 0.25} is reported (ambiguous, nothing output) or, if empty, absent.
  output     frames <= t_a: the forward output; frames > t_a: the committed candidate's backward
             mask where its object score is > 0, nothing if not committed.

Kill criteria, written before the first run (RESULTS Part VII): on the 16 dev episodes with a
baseline, hindsight must produce no switched episode and recover at least 14 (SAM 2.1 alone: 13
recovered, 2 switched, 1 lost; the sound tracker v1: 13 recovered, 0 switched, 2 lost, 1 ambiguous).
Kill: any switched episode, or fewer than 13 recovered.

Not new: forward-backward consistency as a tracking check (Kalal et al. 2010, TLD's
forward-backward error), cycle consistency in SAM 2 variants (SENTRY), offline tracklet stitching
in MOT, SAM 2's own reverse propagation.  What is put together here: the identity decision after an
occlusion made by where jointly back-tracked re-emergence candidates meet the trusted forward track,
with abstention when the meeting is not unique.
"""
from __future__ import annotations

import numpy as np

COMMIT_MEET = 0.5
SECOND_MAX = 0.25
SET_MIN = 0.25


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def first_long_loss(firm: list[bool], run: int = 5) -> int | None:
    """Index of the last firm frame before the first run of >= `run` non-firm frames, or None."""
    n = 0
    for i, f in enumerate(firm):
        n = n + 1 if not f else 0
        if n == run:
            return i - run
    return None


def meet_scores(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    """forward (F, h, w) bool, backward (K, F, h, w) bool -> (K,) median IoU over frames where the forward mask exists."""
    valid = [f for f in range(forward.shape[0]) if forward[f].any()]
    if not valid:
        return np.zeros(backward.shape[0])
    return np.array([float(np.median([iou(forward[f], backward[k, f]) for f in valid])) for k in range(backward.shape[0])])


def decide(m: np.ndarray, commit=COMMIT_MEET, second_max=SECOND_MAX, set_min=SET_MIN) -> dict:
    """{'kind': 'commit'|'ambiguous'|'absent', 'index': int|None, 'set': [indices]}"""
    if m.size == 0:
        return {"kind": "absent", "index": None, "set": []}
    order = np.argsort(-m)
    m1 = float(m[order[0]])
    m2 = float(m[order[1]]) if m.size > 1 else 0.0
    if m1 >= commit and m2 < second_max:
        return {"kind": "commit", "index": int(order[0]), "set": [int(order[0])]}
    s = [int(k) for k in order if m[k] >= set_min]
    return {"kind": "ambiguous" if s else "absent", "index": None, "set": s}
