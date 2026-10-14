#!/usr/bin/env python3
"""P0: the identifiability census of MOT20 occlusion episodes, layer by layer, ground truth only.

    python scripts/census.py                      # all 120 episodes -> out/census.{json,txt}, resumable
    python scripts/census.py --limit 3 --no-anchors
    python scripts/census.py --report             # re-aggregate out/census.json without recomputing

Per episode, from ground-truth boxes and the public detections, no tracker:
  light meter      per-frame gains against the sequence's median background, and their spread
  budget           bin-wise max of the target's per-unit chromaticity histograms over the pre-gap frames
  rarity           which of the target's colours at most 10 % of the nearby crowd shows (confirmation)
  anchors          certified texture patches on the target that certify on no neighbour
  tube             the reachable set through the gap from the other people's boxes; whether the true
                   foot point stayed inside it; nodes, occluders, hand-overs; empty steps
  exit             at the first N_EXIT post-gap frames where the target is half visible: every visible
                   identity in the reach disc gets E (certified over the gain grid), conservation,
                   rho, rare-colour count, anchor certifications; plus the oracle neighbour exclusion
                   (visible >= 80 % of the gap) and the companion distance
The aggregation applies two-fold cross-conformal thresholds (alpha 0.05 per layer) and reports the
layer table disc -> tube -> neighbour -> height -> budget -> confirmation, the hypotheses' measured
violation rates, and the verdict against the kill criteria written in the plan.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crowdtarget import anchors as AN  # noqa: E402
from crowdtarget import episodes as E  # noqa: E402
from crowdtarget import geometry as G  # noqa: E402
from crowdtarget import groups as GR  # noqa: E402
from crowdtarget import identity as ID  # noqa: E402
from crowdtarget import neighbours as NB  # noqa: E402
from crowdtarget import palette as P  # noqa: E402
from crowdtarget import tube as TB  # noqa: E402
from crowdtarget.data.mot import dets_by_frame, frame_path, read_det, read_gt, seq_info, targets  # noqa: E402
from crowdtarget.harness import machine_info, write_out  # noqa: E402

CACHE = ROOT / "data" / "cache"
N_PRE, N_EXIT, EXIT_SPAN = 8, 5, 20
ALPHA = 0.05
V_Q = 0.99
ANCHOR_EXIT_FRAMES, ANCHOR_MAX = 3, 6


def xyxy(rows):
    rows = np.asarray(rows, np.float64).reshape(-1, 9)
    return np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1)


def feet(rows):
    rows = np.asarray(rows, np.float64).reshape(-1, 9)
    return np.stack([rows[:, 2] + rows[:, 4] / 2, rows[:, 3] + rows[:, 5]], 1)


def group_by_frame(gt):
    order = np.argsort(gt[:, 0], kind="stable")
    g = gt[order]
    fr, idx = np.unique(g[:, 0], return_index=True)
    parts = np.split(g, idx[1:])
    return {int(f): p for f, p in zip(fr, parts)}


def speed_bound(gt, q=V_Q):
    sp = []
    for i in np.unique(gt[:, 1]):
        tr = gt[gt[:, 1] == i]
        tr = tr[np.argsort(tr[:, 0])]
        if len(tr) < 10:
            continue
        d = np.hypot(*np.diff(feet(tr), axis=0).T) / np.median(tr[:, 5])
        sp += list(d[np.diff(tr[:, 0]) == 1])
    return float(np.quantile(sp, q))


def det_stats(seq_dir, gt, by_frame, info):
    cache = CACHE / f"{seq_dir.name}_det.json"
    if cache.exists():
        return json.loads(cache.read_text())
    dbf = dets_by_frame(read_det(seq_dir))
    frames = range(1, info["n_frames"] + 1)
    tr = NB.IouTracker()
    tr.run(dbf, frames)
    per_frame = {}
    for tid, t in tr.tracks.items():
        for k, (f, b) in enumerate(zip(t["frames"], t["boxes"])):
            per_frame.setdefault(f, []).append((tid, k, b))
    ids = {tid: [-1] * len(t["frames"]) for tid, t in tr.tracks.items()}
    for f, items in per_frame.items():
        rows = by_frame.get(f, np.zeros((0, 9)))
        g = NB.gt_id_per_detection(np.stack([b for _, _, b in items]), rows)
        for (tid, k, _), gi in zip(items, g):
            ids[tid][k] = int(gi)
    out = {"rule": NB.rule_violations(tr.tracks, ids), "recall": NB.detector_recall(dbf, gt, frames), "n_tracks": len(tr.tracks)}
    cache.write_text(json.dumps(out))
    return out


def seq_assets(seq):
    seq_dir = ROOT / "data" / "mot20" / "MOT20" / "train" / seq
    gt = targets(read_gt(seq_dir))
    info = seq_info(seq_dir)
    CACHE.mkdir(parents=True, exist_ok=True)
    bgp = CACHE / f"{seq}_bg.npy"
    if bgp.exists():
        bg = np.load(bgp)
    else:
        t0 = time.time()
        bg = P.background_model([frame_path(seq_dir, t, info) for t in range(1, info["n_frames"] + 1)])
        np.save(bgp, bg)
        print(f"  background model {seq}: {time.time() - t0:.0f} s", flush=True)
    by_frame = group_by_frame(gt)
    ab, rel = G.height_model(gt)
    t0 = time.time()
    det = det_stats(seq_dir, gt, by_frame, info)
    return {"dir": seq_dir, "gt": gt, "by_frame": by_frame, "info": info, "bg": bg, "ab": ab, "height_rel": rel,
            "v_max": speed_bound(gt), "det": det, "det_seconds": round(time.time() - t0, 1)}


def run_episode(ep, A, use_anchors=True):
    gt, bf, info, bg, ab, vmax = A["gt"], A["by_frame"], A["info"], A["bg"], A["ab"], A["v_max"]
    W, H = info["W"], info["H"]

    def rows_t(t):
        return bf.get(int(t), np.zeros((0, 9)))

    tgt = gt[gt[:, 1] == ep.id]
    pre_rows = tgt[(tgt[:, 0] >= ep.t_start) & (tgt[:, 0] < ep.gap_start) & (tgt[:, 8] >= 0.5)]
    pre_rows = pre_rows[np.argsort(pre_rows[:, 0])]
    if len(pre_rows) < 3:
        return None
    pre_t = pre_rows[np.linspace(0, len(pre_rows) - 1, min(N_PRE, len(pre_rows))).astype(int), 0].astype(int)
    h_ref = float(np.median(pre_rows[:, 5]))
    budget, deltas, fg_fr, anchors = P.Budget(), [], [], []
    others_pres: dict = {}
    n_mined = n_oth_anch = 0
    for k, t in enumerate(pre_t):
        frame = P.load_rgb(frame_path(A["dir"], int(t), info))
        rows = rows_t(t)
        g, d = P.gains(frame, bg, P.boxes_mask(xyxy(rows), H, W))
        deltas.append(float(d.mean()))
        r = rows[rows[:, 1] == ep.id][0]
        box = xyxy(r)[0]
        others = rows[rows[:, 1] != ep.id]
        nearer = xyxy(others[(others[:, 3] + others[:, 5]) > (r[3] + r[5])]) if len(others) else np.zeros((0, 4))
        mask, (x1, y1, x2, y2) = P.foreground_mask(frame, bg, g, box, nearer)
        if mask is None:
            continue
        n, area = int(mask.sum()), (x2 - x1) * (y2 - y1)
        fg_fr.append(n / max(area, 1))
        if n >= 50:
            budget.add(P.histogram(frame[y1:y2, x1:x2].astype(np.float32)[mask] / g, r[5]))
        ov = others[(others[:, 8] >= 0.5) & (others[:, 5] >= 40)]
        order = np.array([], int)
        if len(ov):
            order = np.argsort(np.hypot(*(feet(ov) - feet(r)[0]).T))[:40]
            for o in ov[order]:
                near_o = xyxy(rows[(rows[:, 1] != o[1]) & ((rows[:, 3] + rows[:, 5]) > (o[3] + o[5]))])
                pix_o, n_o, _ = P.foreground(frame, bg, g, xyxy(o)[0], near_o)
                if n_o >= 50:
                    pr = P.presence(P.histogram(pix_o / g, o[5]))
                    others_pres[int(o[1])] = others_pres[int(o[1])] | pr if int(o[1]) in others_pres else pr
        if use_anchors and k in (0, len(pre_t) - 1):
            lum = AN.luminance(frame, g)
            mined = AN.mine(lum, (x1, y1, x2, y2), mask)
            n_mined += len(mined)
            kept, n_o = AN.keep_distinctive(mined, lum, xyxy(ov[order[:10]]) if len(order) else np.zeros((0, 4)))
            anchors += kept
            n_oth_anch += n_o
    if len(budget) == 0:
        return None
    mu = budget.mu
    crowd = list(others_pres.values())
    rare = P.rare_set(P.presence(mu), P.rarity(crowd))
    rare_d = P.dilate_bool(rare)
    crowd_pass = [sum(int((pr & rare_d).sum() >= k) for pr in crowd) for k in (1, 2)]

    # the tube through the gap
    last = tgt[tgt[:, 0] < ep.gap_start]
    last = last[np.argmax(last[:, 0])]
    foot0 = feet(last)[0]
    tb = TB.Tube(W, H)
    tb.start(foot0)
    r_cells = TB.cells_per_frame(vmax, h_ref)
    in_X, n_nodes, area = [], [], []
    for t in range(ep.gap_start, ep.gap_end + 1):
        rows = rows_t(t)
        oth = rows[rows[:, 1] != ep.id]
        tb.step(TB.occluder_grid(xyxy(oth), W, H), r_cells)
        tb.covering(xyxy(oth), oth[:, 1])
        n_nodes.append(tb.n_nodes_now)
        area.append(tb.area_cells)
        tr = rows[rows[:, 1] == ep.id]
        if len(tr):
            in_X.append(tb.contains(feet(tr)[0]))
    tube = {"cells_per_frame": r_cells, "true_in_X_frac": float(np.mean(in_X)) if in_X else float("nan"),
            "true_in_X_last": bool(in_X[-1]) if in_X else None, "empty_steps": tb.empty_steps, "steps": tb.steps,
            "nodes_mean": float(np.mean(n_nodes)), "occluders": len(tb.occluders), "handovers": tb.handovers,
            "area_cells_end": int(area[-1]) if area else 0}

    # exit frames
    post = tgt[(tgt[:, 0] > ep.gap_end) & (tgt[:, 0] <= ep.gap_end + EXIT_SPAN) & (tgt[:, 8] >= 0.5)]
    post = post[np.argsort(post[:, 0])][:N_EXIT]
    if len(post) == 0:
        return None
    exit_t = post[:, 0].astype(int)
    rho_pre = G.person_rho(gt, ep.id, ab, frames=range(ep.t_start, ep.gap_start))
    disc_r = vmax * (ep.g + 10) * h_ref
    comps = GR.companions(gt, ep.id, ep.t_start, ep.gap_start - 1)
    comp_id = comps[0]["id"] if comps else None
    comp_dist = []
    acc = {}
    t_prev = ep.gap_end
    for k_exit, t in enumerate(exit_t):
        for u in range(t_prev + 1, t + 1):            # possibly visible now: the set only grows
            tb.step_free(r_cells)
        t_prev = t
        frame = P.load_rgb(frame_path(A["dir"], int(t), info))
        rows = rows_t(t)
        g, d = P.gains(frame, bg, P.boxes_mask(xyxy(rows), H, W))
        deltas.append(float(d.mean()))
        lum = AN.luminance(frame, g) if (use_anchors and anchors and k_exit < ANCHOR_EXIT_FRAMES) else None
        vis = rows[(rows[:, 8] >= 0.5) & (rows[:, 5] >= 40)]
        comp_foot = None
        if comp_id is not None:
            cr = rows[rows[:, 1] == comp_id]
            comp_foot = feet(cr)[0] if len(cr) else None
        for o, fo in zip(vis, feet(vis)):
            oid = int(o[1])
            in_disc = bool(np.hypot(*(fo - foot0)) <= disc_r)
            in_tube = tb.admits(fo)
            if oid != ep.id and not in_disc:
                continue
            near_o = xyxy(rows[(rows[:, 1] != oid) & ((rows[:, 3] + rows[:, 5]) > (o[3] + o[5]))])
            pix, n, _ = P.foreground(frame, bg, g, xyxy(o)[0], near_o)
            a = acc.setdefault(oid, {"E": [], "Ep": [], "cons": [], "rho": [], "rare": [], "anch": [], "disc": [], "tube": [], "n": []})
            a["disc"].append(in_disc)
            a["tube"].append(in_tube)
            a["n"].append(n)
            a["rho"].append(float(G.rho(o[5], o[3] + o[5], ab)))
            if n >= 50:
                e, cons, e_plain = P.excess_certified(pix, o[5], mu, g, d)
                a["E"].append(e)
                a["cons"].append(cons)
                a["Ep"].append(e_plain)
                a["rare"].append(P.shows(P.histogram(pix / g, o[5]), rare))
            if lum is not None and in_tube:
                a["anch"].append(AN.certified_count(lum, xyxy(o)[0], anchors[:ANCHOR_MAX]))
            if comp_foot is not None and oid == ep.id:
                comp_dist.append(float(np.hypot(*(fo - comp_foot)) / h_ref))

    def agg(oid, a):
        return {"id": oid, "frames": len(a["disc"]), "in_disc": bool(any(a["disc"])), "in_tube": bool(any(a["tube"])),
                "E_min": float(min(a["E"])) if a["E"] else float("nan"), "E_med": float(np.median(a["E"])) if a["E"] else float("nan"),
                "E_plain_min": float(min(a["Ep"])) if a["Ep"] else float("nan"),
                "cons_min": float(min(a["cons"])) if a["cons"] else float("nan"),
                "rho": float(np.median(a["rho"])), "d_rho": float(abs(np.median(a["rho"]) - rho_pre)),
                "rare_max": int(max(a["rare"])) if a["rare"] else 0, "anch_max": int(max(a["anch"])) if a["anch"] else 0,
                "anch_tested": bool(a["anch"]), "vis_gap": NB.oracle_visible_frac(gt, oid, ep.gap_start, ep.gap_end),
                "fg_px_med": float(np.median(a["n"]))}

    cands = [agg(oid, a) for oid, a in acc.items()]
    true = next((c for c in cands if c["id"] == ep.id), None)
    return {"seq": ep.seq, "id": ep.id, "gap_start": ep.gap_start, "gap_end": ep.gap_end, "g": ep.g, "bin": ep.bin,
            "h_ref": h_ref, "n_pre": int(len(budget)), "fg_frac_pre": float(np.mean(fg_fr)) if fg_fr else float("nan"),
            "delta_mean": float(np.mean(deltas)), "mu_total": float(mu.sum()), "n_crowd": len(crowd),
            "n_rare": int(rare.sum()), "crowd_pass1": crowd_pass[0], "crowd_pass2": crowd_pass[1],
            "anchors_mined": n_mined, "anchors_kept": len(anchors), "anchors_others": n_oth_anch,
            "tube": tube, "exit_frames": exit_t.tolist(), "disc_r_heights": float(disc_r / h_ref), "rho_pre": rho_pre,
            "companion": comp_id, "companion_vis_gap": NB.oracle_visible_frac(gt, comp_id, ep.gap_start, ep.gap_end) if comp_id is not None else None,
            "companion_exit_dist": float(np.median(comp_dist)) if comp_dist else None,
            "true": true, "cands": cands}


# ----------------------------------------------------------------------------------------------
def aggregate(records, alpha=ALPHA):
    recs = sorted([r for r in records if r.get("true") is not None], key=lambda r: (r["seq"], r["gap_start"]))
    folds = [recs[0::2], recs[1::2]]
    for i, test in enumerate(folds):
        cal = folds[1 - i]
        thr = {"tau_min": ID.conformal_threshold([r["true"]["E_min"] for r in cal], alpha),
               "tau_med": ID.conformal_threshold([r["true"]["E_med"] for r in cal], alpha),
               "q_cons": ID.conformal_threshold([r["true"]["cons_min"] for r in cal], alpha),
               "kappa": ID.conformal_threshold([r["true"]["d_rho"] for r in cal], alpha)}
        for r in test:
            r["thr"] = thr
            layers = [("disc", lambda c: c["in_disc"]), ("tube", lambda c: c["in_tube"]),
                      ("neighbour", lambda c: c["vis_gap"] < 0.8),
                      ("height", lambda c: c["d_rho"] <= thr["kappa"]),
                      ("budget", lambda c: (not np.isfinite(c["E_min"])) or (c["E_min"] <= thr["tau_min"] and c["cons_min"] <= thr["q_cons"]))]
            res = ID.layered(r["cands"], r["id"], layers)
            S = res["survivors"]
            surv = [c for c in r["cands"] if c["id"] in S]
            conf1 = [c["id"] for c in surv if c["rare_max"] >= 1]
            conf2 = [c["id"] for c in surv if c["rare_max"] >= 2]
            confA = [c["id"] for c in surv if c["anch_max"] >= 1]
            n_crowd = max(r["n_crowd"], 1)
            p1_bar = ID.clopper_pearson_upper(r["crowd_pass1"], n_crowd)
            p2_bar = ID.clopper_pearson_upper(r["crowd_pass2"], n_crowd)
            pA_bar = ID.clopper_pearson_upper(0, max(r["anchors_others"], 1)) if r["anchors_kept"] else 1.0
            true_in = r["id"] in S
            outcome = "excluded@" + str(res["true_dropped_at"]) if not true_in else ("singleton" if len(S) == 1 else "ambiguous")
            commit = None
            if true_in and len(S) > 1:
                for name, conf, pbar in (("anchor", confA, pA_bar), ("rare2", conf2, p2_bar), ("rare1", conf1, p1_bar)):
                    if len(conf) == 1 and len(S) * pbar <= 0.1:
                        commit = (name, conf[0] == r["id"])
                        break
            r["layers"] = res["counts"]
            r["survivors"] = S
            r["outcome"] = outcome
            r["commit"] = commit
            r["false_rare1"] = sum(1 for c in surv if c["id"] != r["id"] and c["rare_max"] >= 1)
            r["false_anchor"] = sum(1 for c in surv if c["id"] != r["id"] and c["anch_max"] >= 1)
            r["p1_bar"], r["p2_bar"], r["pA_bar"] = p1_bar, p2_bar, pA_bar
    return recs


def summarise(recs):
    def block(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        cov = np.mean([r["outcome"] != "excluded" and not r["outcome"].startswith("excluded") for r in rs])
        sizes = [len(r["survivors"]) for r in rs]
        single = np.mean([r["outcome"] == "singleton" for r in rs])
        conf_ok = sum(1 for r in rs if r["commit"] and r["commit"][1])
        conf_bad = sum(1 for r in rs if r["commit"] and not r["commit"][1])
        layers = {k: float(np.median([r["layers"][k] for r in rs])) for k in rs[0]["layers"]}
        drop = {}
        for r in rs:
            if r["outcome"].startswith("excluded@"):
                k = r["outcome"].split("@")[1]
                drop[k] = drop.get(k, 0) + 1
        return {"n": n, "coverage": float(cov), "set_median": float(np.median(sizes)), "set_mean": float(np.mean(sizes)),
                "singleton_correct": float(single), "confirmed_correct": conf_ok, "confirmed_wrong": conf_bad,
                "resolved": float((sum(r["outcome"] == "singleton" for r in rs) + conf_ok) / n),
                "layers_median": layers, "true_dropped_at": drop,
                "E_true_q50": float(np.nanmedian([r["true"]["E_min"] for r in rs])),
                "E_true_q90": float(np.nanquantile([r["true"]["E_min"] for r in rs], 0.9)),
                "E_true_gt_0.3": float(np.mean([r["true"]["E_min"] > 0.3 for r in rs])),
                "true_in_X_mean": float(np.nanmean([r["tube"]["true_in_X_frac"] for r in rs])),
                "true_left_X_last": float(np.mean([r["tube"]["true_in_X_last"] is False for r in rs])),
                "tube_empty_any": float(np.mean([r["tube"]["empty_steps"] > 0 for r in rs])),
                "handovers_mean": float(np.mean([r["tube"]["handovers"] for r in rs])),
                "occluders_mean": float(np.mean([r["tube"]["occluders"] for r in rs])),
                "rare_ge1": float(np.mean([r["n_rare"] >= 1 for r in rs])),
                "true_shows_rare": float(np.mean([r["true"]["rare_max"] >= 1 for r in rs if r["n_rare"] >= 1])) if any(r["n_rare"] >= 1 for r in rs) else float("nan"),
                "false_rare1_mean": float(np.mean([r["false_rare1"] for r in rs])),
                "anchors_kept_mean": float(np.mean([r["anchors_kept"] for r in rs])),
                "true_anchor_certified": float(np.mean([r["true"]["anch_max"] >= 1 for r in rs if r["true"]["anch_tested"]])) if any(r["true"]["anch_tested"] for r in rs) else float("nan"),
                "false_anchor_mean": float(np.mean([r["false_anchor"] for r in rs])),
                "companion": float(np.mean([r["companion"] is not None for r in rs])),
                "companion_dist_q90": float(np.quantile([r["companion_exit_dist"] for r in rs if r["companion_exit_dist"] is not None], 0.9)) if any(r["companion_exit_dist"] is not None for r in rs) else float("nan")}

    out = {"all": block(recs)}
    for b in sorted({r["bin"] for r in recs}):
        out[b] = block([r for r in recs if r["bin"] == b])
    for s in sorted({r["seq"] for r in recs}):
        out[s] = block([r for r in recs if r["seq"] == s])
    return out


def verdict(s):
    a = s["all"]
    kills = []
    if a["coverage"] >= 0.87 and a["set_median"] >= 3:
        kills.append("median set size >= 3 at the coverage achieved")
    if a["true_left_X_last"] > 0.10:
        kills.append(f"true person outside X_t at the end of the gap on {100 * a['true_left_X_last']:.0f} % of episodes (> 10 %)")
    if a["E_true_gt_0.3"] > 0.30:
        kills.append(f"E(target) > 0.3 on {100 * a['E_true_gt_0.3']:.0f} % of episodes (> 30 %)")
    passed = a["coverage"] >= 0.87 and a["set_median"] <= 1 and a["resolved"] >= 0.60
    return {"kills": kills, "pass": bool(passed and not kills)}


def format_report(recs, s, seq_stats, v):
    L = [f"Identifiability census of MOT20 occlusion episodes, ground truth only, {time.strftime('%Y-%m-%d')}", "",
         f"{len(recs)} episodes; thresholds two-fold cross-conformal at alpha {ALPHA} per layer (tau on E_min, q on conservation, kappa on |d rho|).", ""]
    L.append("Per sequence: speed bound v_max (heights/frame), height-model residual, detector recall at vis >= 0.5, continuity-rule violation rate")
    for seq, st in seq_stats.items():
        L.append(f"  {seq}: v_max {st['v_max']:.3f}  height rel-resid {st['height_rel']:.3f}  det recall {st['det']['recall']['recall']:.3f}  "
                 f"rule violated {st['det']['rule']['violated_runs']} of {st['det']['rule']['runs']} certain runs ({100 * st['det']['rule']['run_violation_rate']:.1f} %)")
    L.append("")
    hdr = f"{'group':9s} {'n':>3s} {'cover':>6s} {'|S| med':>7s} {'single':>7s} {'conf+':>5s} {'conf-':>5s} {'resolved':>8s} | {'disc':>5s} {'tube':>5s} {'neigh':>5s} {'height':>6s} {'budget':>6s} | {'E50':>5s} {'E90':>5s} {'E>.3':>5s} | {'inX':>5s} {'leftX':>5s} {'empty':>5s} {'hand':>5s} {'occl':>5s} | {'rare':>5s} {'shows':>5s} {'fRare':>5s} {'anch':>5s} {'aTrue':>5s} {'fAnch':>5s}"
    L.append(hdr)
    for k, b in s.items():
        if b["n"] == 0:
            continue
        lm = b["layers_median"]
        L.append(f"{k:9s} {b['n']:3d} {b['coverage']:6.2f} {b['set_median']:7.1f} {b['singleton_correct']:7.2f} {b['confirmed_correct']:5d} {b['confirmed_wrong']:5d} {b['resolved']:8.2f} | "
                 f"{lm['disc']:5.0f} {lm['tube']:5.0f} {lm['neighbour']:5.0f} {lm['height']:6.0f} {lm['budget']:6.0f} | "
                 f"{b['E_true_q50']:5.2f} {b['E_true_q90']:5.2f} {b['E_true_gt_0.3']:5.2f} | "
                 f"{b['true_in_X_mean']:5.2f} {b['true_left_X_last']:5.2f} {b['tube_empty_any']:5.2f} {b['handovers_mean']:5.1f} {b['occluders_mean']:5.1f} | "
                 f"{b['rare_ge1']:5.2f} {b['true_shows_rare']:5.2f} {b['false_rare1_mean']:5.2f} {b['anchors_kept_mean']:5.1f} {b['true_anchor_certified']:5.2f} {b['false_anchor_mean']:5.2f}")
    L += ["", "Columns: cover = true person in the final set; |S| med = median set size; single = singleton and correct; conf+/- = commits by",
          "confirmation (anchor, then 2 rare colours, then 1) right/wrong; resolved = single + conf+; disc..budget = median candidates left",
          "after each layer; E50/E90 = quantiles of the true person's certified excess at exit; E>.3 = share above 0.3; inX = share of gap",
          "frames with the true foot inside the tube; leftX = share of episodes outside it at the gap's end; empty = tube emptied at least",
          "once; hand = hand-overs; occl = occluders involved; rare = episodes with >= 1 rare colour; shows = true person shows one at",
          "exit; fRare = surviving impostors showing one; anch = distinctive anchors kept; aTrue = true person certified by one; fAnch =",
          "surviving impostors certified.", ""]
    drop = s["all"]["true_dropped_at"]
    L.append("True person dropped by layer: " + (", ".join(f"{k}: {n}" for k, n in drop.items()) if drop else "never"))
    L.append("")
    L.append(f"Verdict against the plan's kill criteria: {'PASS' if v['pass'] else ('KILL: ' + '; '.join(v['kills']) if v['kills'] else 'neither pass nor kill')}")
    L.append("")
    L.append(f"{'episode':24s} {'g':>3s} {'disc':>4s} {'tube':>4s} {'neigh':>5s} {'hgt':>4s} {'bud':>4s} {'outcome':>22s} {'E_true':>6s} {'rare':>4s} {'anch':>4s} {'inX':>4s} {'occl':>4s}")
    for r in recs:
        lm = r["layers"]
        c = f" {r['commit'][0]}{'+' if r['commit'][1] else '-'}" if r["commit"] else ""
        L.append(f"{r['seq'] + '_' + str(r['id']) + '_' + str(r['gap_start']):24s} {r['g']:3d} {lm['disc']:4d} {lm['tube']:4d} {lm['neighbour']:5d} {lm['height']:4d} {lm['budget']:4d} "
                 f"{(r['outcome'] + c):>22s} {r['true']['E_min']:6.2f} {r['true']['rare_max']:4d} {r['true']['anch_max']:4d} {r['tube']['true_in_X_frac']:4.2f} {r['tube']['occluders']:4d}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["all", "dev", "held"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--report", action="store_true", help="aggregate the existing out/census.json only")
    ap.add_argument("--out", default="census")
    a = ap.parse_args()
    pj = ROOT / "out" / f"{a.out}.json"
    old = json.loads(pj.read_text()) if pj.exists() else {"records": [], "seq": {}}
    done = {(r["seq"], r["id"], r["gap_start"]): r for r in old["records"]}
    seq_stats = old.get("seq", {})
    if not a.report:
        eps = E.read(ROOT / "data" / "episodes" / f"mot20_{a.set}.json")
        if a.limit:
            eps = eps[: a.limit]
        assets = {}
        t_all = time.time()
        for k, ep in enumerate(eps):
            key = (ep.seq, ep.id, ep.gap_start)
            if key in done:
                continue
            if ep.seq not in assets:
                assets[ep.seq] = seq_assets(ep.seq)
                A = assets[ep.seq]
                seq_stats[ep.seq] = {"v_max": A["v_max"], "height_rel": A["height_rel"], "det": A["det"], "ab": A["ab"].tolist()}
            t0 = time.time()
            rec = run_episode(ep, assets[ep.seq], use_anchors=not a.no_anchors)
            if rec is None:
                print(f"[{k + 1}/{len(eps)}] {ep.seq} id {ep.id}: skipped (too few usable frames)", flush=True)
                continue
            rec["seconds"] = round(time.time() - t0, 1)
            done[key] = rec
            tr = rec["true"]
            print(f"[{k + 1}/{len(eps)}] {ep.seq} id {ep.id:4d} g {ep.g:3d}: cands {len(rec['cands']):2d}  E_true {tr['E_min']:.2f}  d_rho {tr['d_rho']:.3f}  "
                  f"rare {rec['n_rare']:2d} (true shows {tr['rare_max']}, crowd pass {rec['crowd_pass1']}/{rec['n_crowd']})  anchors {rec['anchors_kept']}/{rec['anchors_mined']} (true cert {tr['anch_max']})  "
                  f"tube inX {rec['tube']['true_in_X_frac']:.2f} empty {rec['tube']['empty_steps']} occl {rec['tube']['occluders']}  {rec['seconds']} s", flush=True)
            payload = {"machine": machine_info(), "records": list(done.values()), "seq": seq_stats, "seconds": round(time.time() - t_all, 1)}
            pj.parent.mkdir(exist_ok=True)
            pj.write_text(json.dumps(payload, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    recs = aggregate(list(done.values()))
    s = summarise(recs)
    v = verdict(s)
    text = format_report(recs, s, seq_stats, v)
    payload = {"machine": machine_info(), "records": recs, "seq": seq_stats, "summary": s, "verdict": v, "alpha": ALPHA}
    write_out(a.out, payload, text)
    print(text)


if __name__ == "__main__":
    main()
