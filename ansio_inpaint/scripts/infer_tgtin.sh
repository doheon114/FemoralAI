#!/bin/bash
# 학습된 G_best.pt 로 split 전체 영상 추론(인과적 anchor prior σ0.05) + 시각화 영상.
# usage: ANSIO_DATA_ROOT=... ANSIO_CLS_CKPT=... GPU=0 bash scripts/infer_tgtin.sh <run_dir> [split=test]
set -e
cd "$(dirname "$0")/.."
: "${ANSIO_DATA_ROOT:?set ANSIO_DATA_ROOT}" "${ANSIO_CLS_CKPT:?set ANSIO_CLS_CKPT}"
export ANSIO_DATA_ROOT ANSIO_CLS_CKPT
RUN=$1; SPLIT=${2:-test}
OUT=$RUN/${SPLIT}_infer_aprior0.05
CUDA_VISIBLE_DEVICES=${GPU:-0} python infer_test.py --ckpt $RUN/G_best.pt --out $OUT --split $SPLIT --full_video --anc_prior 0.05
python viz.py $OUT          # -> ${OUT}_viz/{vid}.mp4
