# Femoral Nerve Visibility Enhancement (COIN-based counterfactual inpainting)

**Status: work in progress / unfinished research.** APIs, configs, and results here are not stable.

## Goal

In ultrasound-guided femoral nerve block, the nerve is sometimes not clearly visible because the probe
is held at an anisotropic angle. This project trains a model that takes a frame where a frozen classifier
says the nerve is `ANISOTROPIC` (angle-dependent, hard to see) and generates a **counterfactual** version of
that same frame as if it had been captured at a `NERVE_VISIBLE` angle — i.e. it enhances nerve visibility
without changing anything else about the frame (probe position, surrounding tissue, `NERVE_NOT_VISIBLE`
frames are left untouched).

This is a counterfactual-explanation problem, not a segmentation/enhancement problem: there is no ground
truth for "what the nerve would have looked like," so the only supervision signal is a frozen 3-class
classifier (`ANISOTROPIC` / `NERVE_NOT_VISIBLE` / `NERVE_VISIBLE`) trained on real frames, plus GAN realism
and minimal-edit losses.

## Method

The generator/discriminator/training loop is adapted from **COIN** (COunterfactual INpainting), a
GAN-based counterfactual explanation method for black-box image classifiers. It is not our own architecture
— credit to the original COIN authors; this repo re-implements and extends their inpainting cGAN for a new
domain (video, not single frames) and a new frozen classifier.

Adaptations made for this domain, on top of vanilla COIN:

- **Frozen classifier is a video model, not a single-frame model.** It is a 3-class ConvNeXt-Tiny + ConvGRU
  network trained with TBPTT (`tools/train_convnext_tbptt.py`) that scores a *window* of `S` consecutive
  frames, carrying hidden state across the window. The GAN's batches are windows `[B, S, 1, H, W]`; the
  generator still runs per-frame (`N = B*S`).
- **Per-frame counterfactual condition.** `c = 1` (inpaint) iff the classifier's argmax for that frame is
  `ANISOTROPIC`; `c = 0` (keep) otherwise. `NERVE_NOT_VISIBLE` frames are never touched.
- **Classifier consistency loss.** Cross-entropy toward `NERVE_VISIBLE` on inpainted frames, KL-divergence
  between `f(x)` and `f(x_cf)` on kept frames (identity should not perturb the classifier's opinion).
- **Real image output.** The generator only emits the grayscale frame; the classifier's actual 3-channel
  input (`[gray, CLAHE, scharr]`) is re-derived from that single output channel, so the counterfactual is a
  real, physically consistent image rather than three independently hallucinated feature maps.
- **Conditional discriminator** on the classifier's 3-way argmax (real frames conditioned on their own
  class, fakes conditioned on the *desired* class — `NERVE_VISIBLE` for inpainted frames, own class for kept
  frames).
- **Temporal modeling**: an optional ConvGRU bottleneck at the encoder's deepest feature map (causal within
  the window) plus a temporal-smoothness loss on the perturbation, so counterfactuals are stable across
  consecutive frames instead of flickering. An EMA copy of the encoder/generator is used for evaluation.
- **Visible-frame memory** (optional, `vis_memory` in config): a space-time-memory read (à la Oh et al.,
  2019, *Video Object Segmentation using Space-Time Memory Networks*) at selected encoder levels. Keys/values
  come from encoder features of past context frames (plus earlier frames in the current window), restricted
  to frames the classifier calls `NERVE_VISIBLE` (hard mask on argmax + a log-p(visible) bias). This lets an
  `ANISOTROPIC` frame copy how the nerve actually looked the last time it was visible in the same video,
  instead of hallucinating an appearance from nothing. An `align` variant additionally registers each memory
  frame onto the current frame (FFT phase-correlation translation) before the read, to counter probe drift.

Evaluation (classifier-driven, no ground-truth masks):

```
flip_pred = P(argmax f(x_cf) == NERVE_VISIBLE | argmax f(x) == ANISOTROPIC)   # counterfactual validity
flip_gt   = P(argmax f(x_cf) == NERVE_VISIBLE | GT == ANISOTROPIC)
keep      = P(argmax f(x_cf) == argmax f(x) | argmax f(x) != ANISOTROPIC)     # identity frames untouched
l1_ani / l1_keep = mean |x_cf - x| on inpainted / kept frames                 # minimality of the edit
score     = flip_pred + keep - l1_ani                                         # model-selection criterion
```

## Repo layout

```
tools/
  train_convnext_tbptt.py   # trains the frozen 3-class video classifier (ConvNeXt-T + ConvGRU, TBPTT)
  verify_femoral_classifier.py  # sanity-checks a trained classifier checkpoint against a config
  train.py                  # trains the counterfactual inpainting GAN (-c <config>, or -cp <run_dir> to resume)
  infer_femoral_cf.py       # runs a trained GAN over whole videos, exports before/after comparison videos + CSV
configs/counterfactual/femoral/
  cf_inpainting_ani2vis.yaml                    # base model, no temporal memory
  cf_inpainting_ani2vis_vismem.yaml             # + visible-frame memory
  cf_inpainting_ani2vis_vismem_align.yaml       # + anchor-aligned visible-frame memory
  cf_inpainting_ani2vis_vismem_align_384.yaml   # same, at 384x384 resolution
src/
  datasets/femoral.py                 # windowed dataset (frame layout below)
  models/tbptt_classifier.py          # frozen classifier + channel derivation ([gray, CLAHE, scharr])
  models/cgan/                        # encoder/generator/discriminator + the femoral cGAN itself
  trainers/counterfactual_inpainting_femoral.py  # training/validation loop + evaluation
```

## Data layout expected

```
<root_dir>/<split>/<CLASS>/<video_id>_f<frame_idx>.png     # 448x448 grayscale
```
`split` is one of `train`/`val`/`test` (video-level split); `CLASS` is one of `ANISOTROPIC`,
`NERVE_NOT_VISIBLE`, `NERVE_VISIBLE`. Frames of one video are scattered across the class folders (each
frame lives under its own per-frame GT label) and reassembled by `video_id` + `frame_idx`. Data itself is
not included in this repo.

## Usage

```bash
pip install -r requirements.txt

# 1. Train the frozen video classifier the GAN will be scored against
python tools/train_convnext_tbptt.py --seq 12 --epochs 40

# 2. Train the counterfactual inpainting GAN (point classifier_ckpt in the config at step 1's output)
PYTHONPATH=$PWD python tools/train.py -c configs/counterfactual/femoral/cf_inpainting_ani2vis_vismem_align.yaml
# resume:
PYTHONPATH=$PWD python tools/train.py -cp training_logs/femoral/<run_dir>

# 3. Export before/after comparison videos for a trained run
PYTHONPATH=$PWD python tools/infer_femoral_cf.py -r training_logs/femoral/<run_dir> --split val
```

## Known limitations / open problems (as of this snapshot)

- GAN training is unstable: the adversarial loss occasionally spikes for a batch or two (LSGAN/MSE loss
  squaring an outlier discriminator output), which can knock the generator into a bad regime for several
  epochs afterward. Model selection (`best.pth`) protects against keeping a collapsed checkpoint, but the
  underlying instability isn't fixed.
- At 384x384 the model is memory- and time-hungry (batch size 1, ~1.9s/it on a 48GB GPU); results at that
  resolution are not yet on par with the 224x224 baseline in this snapshot.
- No quantitative comparison yet against a non-visible-frame-memory baseline beyond flip/keep/l1 metrics on
  a handful of held-out videos.

## Second approach: `ansio_inpaint/` (anchor-windowed, enhance-only edit)

An independent, lighter generator that predicts a small physically constrained brightening edit
(gain / blur-mix at 1/8 resolution) inside one anchor window, adds texture from the most recent real
`NERVE_VISIBLE` frame warped with fixed DIS optical flow, and is trained only through the same frozen video
classifier. Snapshot of the run `a2v_sca_v4_tex_dis_enh_off0_g3_tgtin` (test: 0.752 of `ANISOTROPIC` frames
flipped to `NERVE_VISIBLE`; already-visible frames P(vis) 0.651 → 0.708 with mean |ΔI| 0.003).
See [`ansio_inpaint/README.md`](ansio_inpaint/README.md).

## Credit

Counterfactual inpainting cGAN adapted from **COIN** (COunterfactual INpainting) — this repo is a
domain-specific re-implementation/extension (video classifier, visible-frame memory, temporal modeling) of
their method, not the original COIN codebase.
