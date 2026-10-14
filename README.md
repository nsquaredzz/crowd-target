# crowd-target

One target person through a crowd, on top of a frozen SAM 2.1, with one new piece: a
statement that **the tracker has switched person**, with a controlled false-alarm rate, made
without training and without a noise model.

Status 2026-10-15: scaffold; P0 (data, episodes, SAM 2.1 baseline) in progress. No claim yet.

## The piece

A tracked person gives a stream of nonconformity scores: the distance of the current mask's
appearance embedding to the person's own earlier frames. Online conformal p-values turn that
stream into independent uniforms under exchangeability; a test martingale bets on them; by
Ville's inequality the alarm `W_t >= 1/alpha` has `P(false alarm, ever) <= alpha` under the
null. The mathematics is Vovk's (conformal test martingales, 2003-2021) and Ramdas's
(anytime-valid testing); it has not been used in visual tracking, where switch detection is a
threshold on the model's own score, a cycle-consistency check (SENTRY, 2026) or a learned
verifier. `python/crowdtarget/martingale.py`.

What is measured, not assumed: scores on consecutive frames are correlated, so the empirical
false-alarm rate on episodes without a switch is reported beside alpha, always; the headline is
a ROC against SAM 2's object score and a learned-null CUSUM at matched empirical false-alarm
rate, by occlusion-gap length.

## Protocol

Single-target episodes mined from MOT20 train by fixed rules (`python/crowdtarget/episodes.py`):
an identity with an occlusion gap of 20 to 200 frames, a clean run before and after, a box at
least 48 px tall; bins 20-40 / 40-80 / 80+; a dev set of 8 per bin by sorted stride, the rest
held out. SAM 2.1 base+ is prompted with the ground-truth box 30 frames before the gap and
stepped to 60 frames after it; the outcome is recovered / switched (to whom, when: during the
gap or after) / lost, by mask-in-box containment.

## Reproduce

```
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision numpy pillow scipy pytest hydra-core iopath tqdm
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2@$(cat sam2_commit.txt | cut -d' ' -f3)"
.venv/bin/pip install -e ../certified-skip -e ../certified-track -e .
.venv/bin/python scripts/fetch_data.py                 # MOT20.zip, 5.03 GB, SHA-256 checked
.venv/bin/python scripts/mine_episodes.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_episodes.py --set dev --model b+
```

The SAM 2.1 base+ checkpoint is read from `data/sam2/` or from `../certified-skip/data/sam2/`.
SAM 2 is Meta's code from GitHub at the commit in `sam2_commit.txt` (the `sam2` wheel on PyPI
is a third-party repackaging). `certified-skip` and `certified-track` sit beside this repo for
the renderer, the ffmpeg reader and the output harness.

## What is new, what is not, what is missing

**Not new.** SAM 2.1 and its memory (Meta 2024); training-free memory selection in SAM2Long,
DAM4SAM, SAMURAI, HiM2SAM, SENTRY; single-target episodes as an analysis; mask-in-box
containment; Faster R-CNN box-head features as an appearance descriptor; CUSUM with a learned
null (certified-skip §6); conformal test martingales and anytime-valid testing themselves.

**New, to my knowledge.** The application of a conformal test martingale to a tracker's own
appearance stream as an identity-switch detector with a time-uniform false-alarm guarantee,
the score, the protocol and the measured comparison.

**Missing / not claimed.** Any guarantee when the scores are not exchangeable (consecutive
frames are correlated; the measured false-alarm rate is the honest number); look-alikes whose
appearance does not move at a switch (the martingale is silent; MOT20 has few); a benchmark
number on LaSOT, VOT-LT, LVOS or MOSE (not run); a learned ReID embedding (not used);
everything that would follow the alarm (re-detection, identity sets) is planned, not built.

## Licence

MIT for this code. MOT20 and the SAM 2.1 weights stay under their own terms and are fetched
from their owners' servers, never redistributed here.
