# ansio_inpaint — anchor-windowed, enhance-only ANISOTROPIC → NERVE_VISIBLE editing

**Status: research snapshot.** This is the code used to train
`runs/a2v_sca_v4_tex_dis_enh_off0_g3_tgtin` (the current best-looking variant). It is a second approach,
independent of the COIN-based cGAN in the repo root (`src/`, `tools/`), and shares only the frozen video
classifier (`tools/train_convnext_tbptt.py`) and the data layout.

## Idea

Instead of letting a generator repaint the frame, it predicts a **small, physically constrained edit**
inside one **anchor window** around where it thinks the nerve is, and is trained only through a frozen
3-class classifier (`ANISOTROPIC` / `NERVE_NOT_VISIBLE` / `NERVE_VISIBLE`) plus GAN realism and
minimal-edit losses.

```
out = g + m_eff · ΔI,      m_eff = tissue mask · sigmoid(mask map) · Gaussian(anchor, r)
ΔI  = max(0, gain·(g + α(blur g − g)) + offset − g  +  tex)          # enhance-only
tex = β_hf·(w − blur w) + β_lf·(blur w − blur g),  w = warp(past VISIBLE frame, DIS optical flow)
```

Components (all are `train.py` flags, see `scripts/train_tgtin.sh`):

| flag | what it does |
|---|---|
| `--param physical --up_after 1` | gain / offset / blur-mix / mask predicted at 1/8 resolution (MobileNetV3-S U-Net), non-linearities applied before upsampling — no high-frequency adversarial patterns possible |
| `--anchor_r 0.15` | one anchor = soft-argmax of a heatmap; the edit lives in a Gaussian window around it |
| `--anc_local 2` | soft-argmax only inside the 5x5-cell neighbourhood of the heatmap argmax (a flat heatmap otherwise collapses the anchor to the image centre) |
| `--sca_at 4 8 16` | 2-pass: pass 1 finds the anchor, pass 2 runs Spatial Completion Adapters at three encoder strides with the anchor window as prior |
| `--ref_in 1 --tex 1 --tex_fixed 1` | the most recent past `NERVE_VISIBLE` frame of the same video (≤ 30 frames) is warped onto the current frame with a **fixed** DIS optical flow (learned flow was abused to drag unrelated bright structures into the window) and its texture is added (`β_hf`, `β_lf` learned) |
| `--enhance_only 1` | ΔI ≥ 0 — the edit can only brighten, never erase (STE with 0.1 slope on the negative side) |
| `--gain_range 0.8 3.0 --max_offset 0` | brightness must come from the multiplicative gain (follows existing fibre texture); the additive offset produced a blurry haze |
| `--tgt_in 1` | `NERVE_VISIBLE` frames are also fed (30 % of the batch) with a hinge `P(vis) ≥ 0.8` + `|ΔI|` penalty, so already-visible frames get a small confidence boost and the edit is continuous across class boundaries |
| `--lam_temp 5 --tc_anchor 1` | temporal consistency of residual and anchor between neighbouring frames |

Inference (`infer_test.py --anc_prior 0.05`) runs frames in order and adds the previous frame's anchor as a
Gaussian log-prior (σ = 0.05) to the heatmap — causal, no retraining — which removes anchor jumps.

## Results (this snapshot, 224x224, classifier `convnext_tbptt_best.pt`)

Success = `P(vis) ≥ 0.6` and `P(ani) ≤ 0.2` on the edited `ANISOTROPIC` frames, whole videos, `--anc_prior 0.05`.

| split | ANI success | VIS frames P(vis) in → out | VIS mean \|ΔI\| | anchor jumps > 10 px |
|---|---|---|---|---|
| test (5 videos, 573 ANI frames) | 0.752 | 0.651 → 0.708 | 0.003 | 0.2 % |
| val (4 videos, 456 ANI frames) | 0.669 | 0.618 → 0.681 | 0.004 | 1.2 % |

Known issues: the anchor tends to stay near a video-specific fixed position (narrowing the window to
`r = 0.08` lets it move but drops success to ~0.59); there is no ground truth for the edit, so "success"
only means the frozen classifier is convinced.

## Layout

```
train.py        training loop (G + PatchGAN D + frozen classifier), validation, checkpoints (G_best.pt / last.pt)
infer_test.py   whole-video inference -> per-frame panels, results.csv (probabilities, anchor), mp4
viz.py          re-renders infer_test output: class-probability bars, signed-change overlay, anchor + trail, timeline
models/         generator (LightUNetGenerator: physical params, anchor, SCA, tex, enhance-only), discriminator
data.py         SourceDS: source frames + past reference frame (+ DIS flow) + temporal neighbour
classifier.py   frozen TBPTT classifier wrapper (3-channel [gray, CLAHE, scharr] re-derived from the edited gray)
losses.py, runcfg.py, common.py
scripts/        train_tgtin.sh / infer_tgtin.sh — exact settings of the run above
```

## Usage

```bash
export ANSIO_DATA_ROOT=/path/to/femoral       # {train,val,test}/{ANISOTROPIC,NERVE_NOT_VISIBLE,NERVE_VISIBLE}/<vid>_f<idx>.png (448x448 gray)
export ANSIO_CLS_CKPT=/path/to/convnext_tbptt_best.pt
GPU=0 bash scripts/train_tgtin.sh runs/tgtin                 # ~80 s/epoch on one 48 GB GPU, 100 epochs
GPU=0 bash scripts/infer_tgtin.sh runs/tgtin test            # -> runs/tgtin/test_infer_aprior0.05_viz/<vid>.mp4
```

Optional: precomputed scharr images under `$ANSIO_DATA_ROOT/femoral_scharr/` (same relative paths) speed
up loading; otherwise they are computed on the fly. `ANSIO_SIZE` changes the generator resolution
(default 224). Requires `opencv-python` (DIS optical flow), `torch`, `torchvision`.
The SAM2-based options in `train.py` (`--s2m`, `infer_test.py --sam2g`) need `models/sam2mem.py` /
`models/sam2g.py`, which are not part of this snapshot.
