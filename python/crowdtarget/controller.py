"""The sound tracker, first cut (plan step 2): SAM 2.1 as the engine, its commits vetoed by certificates.

Per frame SAM 2 proposes a mask.  The mask is *accepted* (kept in memory, output) only if it passes:
  own gates     object score > 0 and mask area >= `area_gate` x the running median accepted area;
  budget        certified excess E <= tau and conservation <= q against the person's own colour budget,
                learned only on confident frames while the mask sits on the person's own detection track;
  known others  the mask's box does not coincide (IoU >= 0.5) with the current box of a *known other*: a
                detection track that stood beside the person (box IoU with the accepted mask < 0.3)
                while the person was being tracked, and has stayed on a continuous, mostly certain
                track since.  Humans: "that is the man who was standing next to him".  Coincidence,
                not containment: a fragment of the person seen behind an occluder lies inside the
                occluder's box and must not be refused (the first dev run refused id 85 that way).
  reach         while absent, the mask's foot point lies in the tube's reach.
A rejected frame is dropped from SAM 2's memory and the last accepted frames are re-aliased to the six
indices SAM 2 reads (as SAMURAI does), so the occluder is never remembered.  The first rejection starts
the tube from the last firm foot point; it propagates through the detections' boxes while absent.

While absent, the candidates are SAM 2's own mask if it passes everything, plus every detection in the
reach that is not a known other and passes the budget.  Exactly one candidate is a *provisional* commit
(a detection re-prompts SAM 2 with its box); it becomes firm after `firm_after` consecutive accepted
frames and is *revoked* if it fails before that: the re-prompt's conditioning frame is removed from
SAM 2's memory and the state returns to absent with the tube untouched.  Two or more candidates is
ambiguous (the set is logged, nothing committed).  None is absent.  Output: the accepted mask, or empty.

Re-acquisition gate (added after the first dev run, on seeing id 505 accept a fragment of a hidden
neighbour at score 0.8 covering 18 % of the person's mass): while not firmly tracking, a return, by the
mask or by a re-prompt, needs SAM 2 object score > 2 and conservation >= 0.4, i.e. a confident mask
showing at least 40 % of the person.  Judged on the held-out episodes, not on dev.

Lightweight hypotheses (added after the first YOLO dev run, on seeing id 169 lose the person who had
re-emerged next to the tube and then walked out of its reach while the tracker was busy re-prompting a
night-time false detection 54 times): a detection track that entered the reach while absent stays a
candidate for as long as its track lives, so the reach is only the gate of entry; a candidate needs
detector confidence >= 0.25 and a track at least 3 frames old; a track whose prompt SAM 2 refused or whose
provisional commit was revoked is not re-prompted for 10 frames (it stays a hypothesis).  This is the
plan's second tier of hypotheses, carried by detections instead of SAM 2 forks.

Certified sliver chains (`use_chain`, RESULTS Part VI): templates of the person's own texture are mined on
every accepted mask; on the next frame their certified positions (box test, exact offset, 0 false steps on
neighbours in 1,288 census trials) say where the person is regardless of how little of them shows.  A
small mask that contains the certified positions is accepted (the area gate is waived); a mask that
misses them while they exist makes SAM 2 re-prompt on this frame with positive points at the certified
positions and negative points on nearby known others ("that is him, that is not"); the result is kept
if it contains the positions, passes the budget and is not a known other.  A blackout widens the
chain's search at walking speed; the tube and detections remain the fallback.

Not in this cut (plan P2 full): forks per hypothesis, checkpoints further back than the last commit,
confirmation by rare colours or anchors, companions.  Thresholds come from the census (tau = 0.2, the
larger of its two cross-fold values; q = 1.1) and are stated, not tuned on outcomes.
"""
from __future__ import annotations

import numpy as np

from . import neighbours as NB
from . import palette as P
from . import sam2io
from . import sliver as SL
from . import tube as TB
from .fixes import Fix, box_iou, mask_box

TRACKING, ABSENT, AMBIGUOUS = "tracking", "absent", "ambiguous"


def containment(mask: np.ndarray, box) -> float:
    n = int(mask.sum())
    if n == 0:
        return 0.0
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    H, W = mask.shape
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].sum() / n)


class Sound(Fix):
    name = "sound"

    def __init__(self, pred, W, H, start_box, *, bg, dets, v_max, frames, t_start, tau=0.2, q_cons=1.1, area_gate=0.5,
                 cont_frac=0.6, conf=0.1, n_mem=6, firm_after=5, other_iou=0.3, other_iou_veto=0.5, reacq_score=2.0, reacq_cons=0.4,
                 cand_conf=0.25, min_track_age=3, cooldown_frames=10, use_chain=False, chain_max_n=8, chain_search_r=10, jump_iou=0.4):
        super().__init__(pred, W, H)
        self.bg, self.dets, self.v_max, self.frames, self.t_start = bg, dets, v_max, frames, t_start
        self.tau, self.q_cons, self.area_gate, self.cont_frac, self.conf, self.n_mem = tau, q_cons, area_gate, cont_frac, conf, n_mem
        self.firm_after, self.other_iou, self.other_iou_veto = firm_after, other_iou, other_iou_veto
        self.reacq_score, self.reacq_cons = reacq_score, reacq_cons
        self.cand_conf, self.min_track_age, self.cooldown_frames = cand_conf, min_track_age, cooldown_frames
        self.use_chain = use_chain
        self.jump_iou = jump_iou
        self.last_bb = None
        self.chain = SL.Chain(max_n=chain_max_n, search_r=chain_search_r) if use_chain else None
        self.name = "sliver" if use_chain else "sound"
        self.tracker = NB.IouTracker(max_miss=2)
        self.budget = P.Budget()
        self.state = TRACKING
        self.firm = True                       # the current TRACKING run is firm (>= firm_after accepted frames)
        self.run_len = 0
        self.provisional = None                # {"frame": i, "kind": "det"|"mask"} while a commit is on probation
        self.bank: list[tuple[int, dict]] = []
        self.areas: list[float] = []
        self.heights: list[float] = []
        self.own_track: int | None = None
        self.others: set[int] = set()
        self.other_last: dict[int, tuple[int, np.ndarray]] = {}   # tid -> (last frame, last box) of known others
        self.absent_since: int | None = None
        self.last_foot = np.array([(start_box[0] + start_box[2]) / 2, start_box[3]], float)
        self.last_box_h = float(start_box[3] - start_box[1])
        self.tube = TB.Tube(W, H)
        self.tube_live = False
        self.hyps: set[int] = set()                 # detection tracks that entered the reach while absent (lightweight hypotheses)
        self.cooldown: dict[int, int] = {}          # track id -> frame until which it is not re-prompted (its prompt was revoked)
        self.log: list[dict] = []
        self.out_mask = None
        self._g = self._d = None

    # --- plumbing ---------------------------------------------------------------------------------
    def _dets(self, t):
        b, c = self.dets.get(int(t), (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
        keep = np.asarray(c) >= self.conf
        return np.asarray(b, np.float64).reshape(-1, 4)[keep], np.asarray(c)[keep]

    def _h_ref(self):
        return float(np.median(self.heights[-30:])) if self.heights else self.last_box_h

    def _budget_ok(self, pix, h):
        if len(self.budget) < 3 or pix.shape[0] < 50:
            return True, float("nan"), float("nan")
        e, cons, _ = P.excess_certified(pix, h, self.budget.mu, self._g, self._d)
        return (e <= self.tau and cons <= self.q_cons), e, cons

    def _is_known_other(self, tid) -> bool:
        """A track that stood beside the person and has been continuous and mostly certain since absence began."""
        if tid is None or tid not in self.others or tid == self.own_track:
            return False
        tr = self.tracker.tracks.get(tid)
        if tr is None:
            return False
        if self.absent_since is None:
            return True                                   # while tracking, a known other is simply a known other
        since = self.absent_since
        return NB.certain_frac(tr, since, self.cur) >= self.cont_frac or self.cur - since < 2

    def _relink_others(self, i, boxes):
        """A track that starts this frame on top of a known other's last box (ended <= 3 frames ago) is that other."""
        for j, b in enumerate(boxes):
            tid = self.tracker.last_assignment.get(j)
            if tid is None or tid in self.others:
                continue
            tr = self.tracker.tracks.get(tid)
            if tr is None or tr["frames"][0] != i:
                continue
            for ot, (tf, ob) in list(self.other_last.items()):
                if i - tf <= 3 and box_iou(b, ob) >= 0.5:
                    self.others.add(tid)
                    break
        for j, b in enumerate(boxes):
            tid = self.tracker.last_assignment.get(j)
            if tid in self.others:
                self.other_last[tid] = (i, b)

    def _track_of_box(self, box, boxes):
        if boxes.shape[0] == 0:
            return None
        ious = np.array([box_iou(box, b) for b in boxes])
        j = int(np.argmax(ious))
        return self.tracker.last_assignment.get(j) if ious[j] >= 0.5 else None

    def _other_boxes(self, boxes):
        out = []
        for j, b in enumerate(boxes):
            if self._is_known_other(self.tracker.last_assignment.get(j)):
                out.append(b)
        return out

    # --- hooks ------------------------------------------------------------------------------------
    def before_step(self, state, i):
        self.active = True
        self.cur = i
        boxes, conf = self._dets(self.t_start + i)
        self.cur_boxes, self.cur_conf = boxes, conf
        self.tracker.step(i, boxes)
        self._relink_others(i, boxes)
        f = self.frames[i]
        self._g, self._d = P.gains(f, self.bg, P.boxes_mask(boxes, self.H, self.W))
        self._lum = SL.luminance(f, self._g) if self.use_chain else None
        d = state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        if self.bank:
            d.clear()
            for r, (_, out) in enumerate(reversed(self.bank[-self.n_mem:]), 1):
                d[i - r] = out
        if self.tube_live:
            self.tube.step(TB.occluder_grid(boxes, self.W, self.H), TB.cells_per_frame(self.v_max, self._h_ref()))

    def _reject(self, state, i):
        d = state["output_dict_per_obj"][0]
        d["non_cond_frame_outputs"].pop(i, None)
        if self.provisional is not None:
            # revoke: the re-prompt's conditioning frame leaves SAM 2's memory
            pf = self.provisional["frame"]
            if self.provisional["kind"] == "det":
                d["cond_frame_outputs"].pop(pf, None)
                d["non_cond_frame_outputs"].pop(pf, None)
                state["point_inputs_per_obj"][0].pop(pf, None)
                state["mask_inputs_per_obj"][0].pop(pf, None)
            self.bank = [(k, o) for k, o in self.bank if k < pf]
            if self.provisional["kind"] == "det" and getattr(self, "last_prompt_track", None) is not None:
                self.cooldown[self.last_prompt_track] = i + self.cooldown_frames
            self.provisional = None
        if self.state == TRACKING and self.firm:
            self.tube.start(self.last_foot)
            self.tube_live = True
            self.absent_since = i
        self.state = ABSENT
        self.firm = False
        self.run_len = 0

    def _accept(self, state, i, mask, bb, score, boxes, kind):
        d = state["output_dict_per_obj"][0]
        if i in d["non_cond_frame_outputs"]:
            self.bank.append((i, d["non_cond_frame_outputs"][i]))
        elif i in d["cond_frame_outputs"]:
            self.bank.append((i, d["cond_frame_outputs"][i]))
        self.bank = self.bank[-self.n_mem:]
        self.run_len += 1
        if self.state != TRACKING:
            self.provisional = {"frame": i, "kind": kind} if self.provisional is None else self.provisional
        self.state = TRACKING
        if not self.firm and self.run_len >= self.firm_after:
            self.firm = True
            self.provisional = None
            self.absent_since = None
            self.tube_live = False
        self.last_bb = bb if bb is not None else getattr(self, "last_bb", None)
        if self.firm and bb is not None:
            a = float(mask.sum())
            if not self.areas or a >= self.area_gate * float(np.median(self.areas[-30:])):
                self.areas.append(a)
                self.heights.append(float(bb[3] - bb[1]))
            self.last_foot = np.array([(bb[0] + bb[2]) / 2, bb[3]], float)
            self.last_box_h = float(bb[3] - bb[1])
            tid = self._track_of_box(bb, boxes)
            if self.own_track is None and tid is not None:
                self.own_track = tid
            elif tid is not None and tid != self.own_track:
                self.own_track = tid              # the person's own track was re-linked by the detector
            # every other simultaneous detection is another person, companions included; a duplicate detection of
            # the person (IoU >= 0.6 with the person's own detection) is not
            own_box = None
            if tid is not None:
                for j, b in enumerate(boxes):
                    if self.tracker.last_assignment.get(j) == tid:
                        own_box = b
            ref = own_box if own_box is not None else bb
            for j, b in enumerate(boxes):
                tj = self.tracker.last_assignment.get(j)
                if tj is not None and tj != self.own_track and box_iou(b, ref) < 0.6:
                    self.others.add(tj)
                    self.other_last[tj] = (i, b)
            if score > 2 and (not self.areas or a >= self.area_gate * float(np.median(self.areas[-30:]))):
                self.budget.add(P.histogram(self.frames[i][mask].astype(np.float32) / self._g, bb[3] - bb[1]))

    def after_step(self, state, i, mask, score):
        self.active = False
        boxes = self.cur_boxes
        bb = mask_box(mask)
        area = float(mask.sum())
        rec = {"state": self.state, "score": float(score), "area": area, "firm": self.firm}
        reacq = not (self.state == TRACKING and self.firm)       # re-acquisition: a return needs a confident, substantial mask
        # certified sliver chain: where the person's visible texture from the last accepted frame is now
        found = self.chain.probe(self._lum) if (self.use_chain and self.chain.templates) else []
        chained = bool(found) and bb is not None and self.chain.inside(found, mask)
        rec["chain_found"] = len(found)
        rec["chained"] = bool(chained)
        big_enough = (not self.areas) or area >= self.area_gate * float(np.median(self.areas[-30:]))
        # without a certified texture step, the mask must at least overlap the last accepted one (a jump to a
        # neighbour does not); with one, the chain decides
        prev_bb = getattr(self, "last_bb", None)
        continuous = chained or prev_bb is None or (bb is not None and box_iou(bb, prev_bb) >= self.jump_iou) or self.state != TRACKING
        rec["continuous"] = bool(continuous)
        own_ok = bb is not None and score > 0 and (big_enough or chained) and (chained or not reacq or score > self.reacq_score) and continuous
        other_boxes = self._other_boxes(boxes)        # also while firmly tracking: a slide onto a neighbour happens there
        # the mask IS a known other when their boxes coincide; a fragment of the person seen behind that
        # other sits inside the other's box but has a small IoU with it, and must not be vetoed
        # ... or the mask sits almost entirely in a known other's box and not where the person last was
        prev_bb0 = getattr(self, "last_bb", None)
        veto = bb is not None and any(box_iou(bb, ob) >= self.other_iou_veto or
                                      (containment(mask, ob) >= 0.8 and (prev_bb0 is None or containment(mask, prev_bb0) < 0.5))
                                      for ob in other_boxes)
        e = cons = float("nan")
        bud_ok = False
        if own_ok and not veto:
            h = (bb[3] - bb[1]) if (self.state == TRACKING and self.firm) else max(bb[3] - bb[1], self._h_ref())
            bud_ok, e, cons = self._budget_ok(self.frames[i][mask].astype(np.float32), h)
            if reacq and cons == cons and cons < self.reacq_cons and not chained:
                bud_ok = False
        in_reach = (not self.tube_live) or (bb is not None and self.tube.admits(((bb[0] + bb[2]) / 2, bb[3])))
        mask_ok = own_ok and bud_ok and not veto and in_reach
        rec.update({"E": e, "cons": cons, "veto": bool(veto), "own_ok": bool(own_ok), "in_reach": bool(in_reach), "n_others": len(other_boxes),
                    "max_other_contain": float(max([containment(mask, ob) for ob in other_boxes], default=0.0)) if bb is not None else 0.0,
                    "max_other_iou": float(max([box_iou(bb, ob) for ob in other_boxes], default=0.0)) if bb is not None else 0.0,
                    "prev_contain": float(containment(mask, prev_bb0)) if (bb is not None and prev_bb0 is not None) else float("nan")})
        if not mask_ok and found and self.use_chain:
            # SAM 2 missed or misplaced the person, the certified sliver did not: point it there, and away from known others
            cy, cx = SL.centre(found)
            pos = [[f["x"], f["y"]] for f in sorted(found, key=lambda f: f["score"])[:3]]
            neg = []
            for ob in self._other_boxes(boxes) if other_boxes else []:
                ocx, ocy = (ob[0] + ob[2]) / 2, ob[1] + 0.35 * (ob[3] - ob[1])
                if np.hypot(ocx - cx, ocy - cy) <= 1.5 * self._h_ref() and not any(ob[0] <= p[0] <= ob[2] and ob[1] <= p[1] <= ob[3] for p in pos):
                    neg.append([ocx, ocy])
            od = state["output_dict_per_obj"][0]
            od["non_cond_frame_outputs"].pop(i, None)
            sam2io.prompt_points(self.pred, state, i, 1, pos + neg[:3], [1] * len(pos) + [0] * len(neg[:3]))
            self.pred.propagate_in_video_preflight(state)
            out = od["cond_frame_outputs"].get(i) or od["non_cond_frame_outputs"].get(i)
            if out is not None:
                import torch
                with torch.inference_mode():
                    m = torch.nn.functional.interpolate(out["pred_masks"].to(self.pred.device).float(), size=(self.H, self.W), mode="bilinear", align_corners=False)
                m2 = (m[0, 0] > 0).cpu().numpy()
                bb2 = mask_box(m2)
                score2 = float(out["object_score_logits"].reshape(-1)[0])
                ok2, e2, c2 = self._budget_ok(self.frames[i][m2].astype(np.float32), max(bb2[3] - bb2[1], self._h_ref()) if bb2 is not None else 1.0) if bb2 is not None else (False, float("nan"), float("nan"))
                veto2 = bb2 is not None and any(box_iou(bb2, ob) >= self.other_iou_veto for ob in other_boxes)
                if bb2 is not None and score2 > 0 and ok2 and not veto2 and self.chain.inside(found, m2):
                    mask, bb, score = m2, bb2, score2
                    mask_ok = True
                    rec["chain_prompt"] = {"pos": len(pos), "neg": len(neg[:3]), "score": score2, "E": e2}
                else:
                    od["cond_frame_outputs"].pop(i, None)
                    od["non_cond_frame_outputs"].pop(i, None)
                    state["point_inputs_per_obj"][0].pop(i, None)
                    rec["chain_prompt_refused"] = {"score": score2, "E": e2, "inside": bool(bb2 is not None and self.chain.inside(found, m2))}
        if mask_ok:
            self._accept(state, i, mask, bb, score, boxes, "chain" if (chained and rec["state"] != TRACKING) else "mask")
            rec["commit"] = ("chain" if chained else "mask") if rec["state"] != TRACKING else None
            if self.use_chain:
                if found:
                    self.chain.commit(found, self._lum, mask)
                elif continuous:
                    self.chain.commit([], self._lum, mask)        # (re)seed from the accepted mask
        else:
            if self.use_chain:
                self.chain.miss()
            self._reject(state, i)
            # candidates while absent: detections in the reach, not known others, passing the budget
            cands = []
            live = set(self.tracker.last_assignment.values())
            self.hyps &= live                                    # a hypothesis dies with its track
            for k, b in enumerate(boxes):
                ktid = self.tracker.last_assignment.get(k)
                if ktid is None:
                    continue
                in_reach_k = (not self.tube_live) or self.tube.admits(((b[0] + b[2]) / 2, b[3]))
                if not in_reach_k and ktid not in self.hyps:
                    continue
                if self._is_known_other(ktid):
                    continue
                tr = self.tracker.tracks.get(ktid)
                if tr is None or len(tr["frames"]) < self.min_track_age or (self.cur_conf[k] if k < len(self.cur_conf) else 1.0) < self.cand_conf:
                    continue
                nearer = boxes[boxes[:, 3] > b[3] + 1]
                pix, n, _ = P.foreground(self.frames[i], self.bg, self._g, b, nearer)
                ok, eb, cb = self._budget_ok(pix, max(b[3] - b[1], 1.0))
                if ok:
                    self.hyps.add(ktid)                          # once a candidate, carried by its track
                    if self.cooldown.get(ktid, -1) >= i:
                        continue                                 # its prompt was revoked recently: still a hypothesis, not re-prompted yet
                    cands.append((b, k, eb, ktid))
            rec["n_cands"] = len(cands)
            rec["cands"] = [np.asarray(c[0]).round(1).tolist() for c in cands]
            rec["n_hyps"] = len(self.hyps)
            if len(cands) == 1:
                b, k, eb, ktid = cands[0]
                self.last_prompt_track = ktid
                sam2io.prompt_box(self.pred, state, i, 1, b)
                self.pred.propagate_in_video_preflight(state)
                od = state["output_dict_per_obj"][0]
                out = od["cond_frame_outputs"].get(i) or od["non_cond_frame_outputs"].get(i)
                if out is not None:
                    import torch
                    with torch.inference_mode():
                        m = torch.nn.functional.interpolate(out["pred_masks"].to(self.pred.device).float(), size=(self.H, self.W), mode="bilinear", align_corners=False)
                    mask = (m[0, 0] > 0).cpu().numpy()
                    bb = mask_box(mask)
                    score_p = float(out["object_score_logits"].reshape(-1)[0])
                    ok_p, e_p, c_p = self._budget_ok(self.frames[i][mask].astype(np.float32), max(b[3] - b[1], 1.0)) if bb is not None else (False, float("nan"), float("nan"))
                    if bb is not None and score_p > self.reacq_score and ok_p and not (c_p == c_p and c_p < self.reacq_cons):
                        self._accept(state, i, mask, bb, score_p, boxes, "det")
                        rec["commit"] = "det"
                        rec["commit_E"] = eb
                    else:
                        # SAM 2 does not see a whole person of the right colours there: undo the prompt
                        od["cond_frame_outputs"].pop(i, None)
                        od["non_cond_frame_outputs"].pop(i, None)
                        state["point_inputs_per_obj"][0].pop(i, None)
                        rec["refused_det"] = {"score": score_p, "E": e_p, "cons": c_p}
                        self.cooldown[ktid] = i + self.cooldown_frames
                        self.state = ABSENT
                        mask = np.zeros_like(mask)
                else:
                    mask = np.zeros_like(mask)
            else:
                self.state = AMBIGUOUS if len(cands) >= 2 else ABSENT
                mask = np.zeros_like(mask)
        rec["accepted"] = self.state == TRACKING
        rec["output"] = bool(self.state == TRACKING and self.firm) or (self.state == TRACKING and rec.get("commit") is None and self.provisional is None)
        if self.state == TRACKING and not self.firm:
            rec["provisional"] = True
        rec["state_after"] = self.state
        rec["firm_after"] = bool(self.firm and self.state == TRACKING)
        self.log.append(rec)
        self.out_mask = mask if self.state == TRACKING else np.zeros_like(mask)
        return rec
