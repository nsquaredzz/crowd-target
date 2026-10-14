# Results

Every number is in a file under `out/` with the machine and timings. Mac M4, 16 GB, MPS. MOT20
train, single-target episodes mined by the rules in `python/crowdtarget/episodes.py`.

## Part I — the data regime (2026-10-15)

Census of occlusion gaps in MOT20 train (`out/census_mot20.txt`): 2,064 maximal runs of 20
frames or more at visibility < 0.2, but under the v1 episode rules only 8 episodes. The rule
that removes almost everything is the clean run before and after the gap: the median box
visibility in these crowds is 0.35-0.52, so a full occlusion is rarely bracketed by half-visible
frames. v2 rules (gap at visibility < 0.3, 15 of 30 pre-gap and 40 of 60 post-gap frames at
>= 0.5, height >= 40 px), set before any tracker ran: 120 episodes, 77 / 29 / 14 in the gap
bins 20-40 / 40-80 / 80+, dev set 24 (8 per bin by sorted stride), 96 held out
(`out/episodes_mot20.txt`).

## Part II — Piece 1, the conformal switch detector, closed (2026-10-15)

The mathematics holds: on simulated exchangeable streams the online conformal p-values are
uniform (KS distance < 0.02 over 8,000 draws) and both martingales keep the false-alarm rate
under Ville's bound over 2,000 streams at alpha 0.1 and 0.05, while a shift of 2.5 standard
deviations is caught in a median of under 25 steps (`tests/test_martingale.py`).

What fails is the premise that a tracked person's appearance after an occlusion gap is
exchangeable with its appearance before. Measured on ground-truth crops of the 24 dev
episodes, no tracker involved (`out/embed_gap_check_dev*.txt`): the pool is the target's crops
before the gap (the 30-frame pre window, or 300 frames of history), scored by 1 - cosine to the
pool's mean embedding; "same" is the target's crops after the gap, "other" every other
identity in those frames.

| embedding | pool | same-person post-gap crops above the whole pool | AUC same-across-gap vs other, pooled | median per episode |
|---|---|---|---|---|
| Faster R-CNN box head | 30 pre-gap frames | 96 % | 0.69 | 0.69 |
| Faster R-CNN box head | 300 frames | 61 % | 0.68 | 0.74 |
| DINOv2 ViT-S/14 | 30 pre-gap frames | 89 % | 0.75 | 0.81 |
| DINOv2 ViT-S/14 | 300 frames | 45 % | 0.71 | 0.81 |

Without a gap the same box-head feature separates identities at AUC 0.90 (frames 15-45 apart,
`out/embed_check.txt`); a half-occluded crop of the same person sits at distance 0.21, between
same (0.12) and other (0.27). Across a gap, per episode, the DINOv2 share above the pool is
bimodal: 0 % on ten episodes, 90-100 % on nine. On the latter the person has turned, moved
into other light or changed what is visible of them; their new appearance is an outlier to
their own history, and a conformal p-value against that history is ~1/(n+1) whatever the
identity. A test martingale on such a stream detects appearance change, not identity change,
and alarms on about half the episodes in which SAM 2 recovers the right person. The first
tracked episode shows it (`out/p1_p0_sam2_bp_dev*.txt`, `MOT20-01_42_256`): recovered, every
frame on the right person, wealth 500 after the gap under every gate tried.

Verdict against the plan's kill criterion (false alarms above 2 alpha at k = 12): failed, by
construction of the score rather than by the test, on both embeddings available without
training. Not pursued further: a learned ReID network (would narrow the across-gap gap but not
close it on the 90-100 % episodes, where the appearance did change), masking the background
with SAM 2's mask (changed nothing consistently on two episodes), testing the post-gap stream
against itself (answers a different question).

What survives: the census (the clean-occlusion regime is rare in MOT20), the across-gap
separability tables (a number people who build ReID-based re-association on crowds may want),
and a validated implementation of conformal test martingales (`python/crowdtarget/martingale.py`).

## Part III — SAM 2.1 base+ alone, and the martingale on its stream (16 of 24 dev episodes)

The run was stopped by decision after the 20-40 and 40-80 bins, once Part II had settled the
question; the 80+ bin did not run. `scripts/run_episodes.py --set dev`, 1.5-2.6 s per frame at
1080p on MPS, prompted with the ground-truth box 30 frames before the gap, memory as shipped.

| bin | n | recovered | switched (in gap / after) | lost |
|---|---|---|---|---|
| 20-40 | 8 | 7 | 1 (0 / 1) | 0 |
| 40-80 | 8 | 6 | 1 (1 / 0) | 1 |
| all | 16 | 13 | 2 | 1 |

`out/p0_sam2_bp_dev.txt`. SAM 2.1 alone recovers 13 of 16: the failure the project set out to
attack is rare on these episodes (the plan's own P0 threshold was switched + lost below 15 %,
here 19 % on a small sample). One switch began inside the gap, a drift onto the occluder.

The martingale on the tracked stream, gated to frames where SAM 2 claims a whole person
(object score > 2, mask area >= half the pre-gap median), `out/p1_p0_sam2_bp_dev_gs2_ga0.5.txt`:

| detector, alpha 0.05 | stream | switches detected within 30 frames | recovered episodes alarmed |
|---|---|---|---|
| mixture martingale | every 4th frame | 0 of 2 | 13 of 13 (one more alarmed before its switch) |
| simple jumper | every 4th frame | 1 of 2 | 13 of 13 |
| learned-null CUSUM, h = 6 | every 4th frame | 0 of 2 | 13 of 13 |
| mixture martingale | every 12th frame | 0 of 2 | 12 of 13 |
| simple jumper | every 12th frame | 0 of 2 | 0 of 13 |
| SAM 2 object score < 2 | every frame | 0 of 2 | 3 of 13 |

As Part II predicted from the crops: on the tracked stream the martingales alarm on every
recovered episode at every alpha, and the one setting that does not alarm (the jumper at every
twelfth frame) detects nothing. Proof videos: `out/proof_recovered_MOT20-01_42.mp4` (recovered,
wealth rising through the gap and after it) and `out/proof_switch_MOT20-05_611.mp4` (the switch
inside the gap).
