#!/bin/bash
# runs/a2v_sca_v4_tex_dis_enh_off0_g3_tgtin 재현 (처음부터 100 epoch, 224x224).
#   ANSIO_DATA_ROOT : {train,val,test}/{ANISOTROPIC,NERVE_NOT_VISIBLE,NERVE_VISIBLE}/<vid>_f<idx>.png 루트
#   ANSIO_CLS_CKPT  : 고정 3-class 분류기 (ConvNeXt-T + ConvGRU TBPTT, tools/train_convnext_tbptt.py 출력)
# usage: ANSIO_DATA_ROOT=/data/femoral ANSIO_CLS_CKPT=/ckpt/convnext_tbptt_best.pt GPU=0 bash scripts/train_tgtin.sh [out_dir]
set -e
cd "$(dirname "$0")/.."
: "${ANSIO_DATA_ROOT:?set ANSIO_DATA_ROOT}" "${ANSIO_CLS_CKPT:?set ANSIO_CLS_CKPT}"
export ANSIO_DATA_ROOT ANSIO_CLS_CKPT
OUT=${1:-runs/a2v_sca_v4_tex_dis_enh_off0_g3_tgtin}
CUDA_VISIBLE_DEVICES=${GPU:-0} python train.py --lam_gan 1.0 --lam_rec 5.0 --lam_tv 2.0 --lam_area 3.0 --lam_compact 0.0 --area_target 0.06 --beta3 0.5 --gan_mode lsgan --batch 16 --lr_g 0.0002 --lr_d 0.0002 --max_delta 1.0 --anchor_r 0.15 --param physical --repeat 1 --eot 1 --cls_input rederive --d_ch 1 --max_gap 5 --direction ani2vis --target_p 0.6 --src_p 0.2 --gain_range 0.8 3.0 --max_offset 0 --alpha_range -1.0 0.5 --sca_at 4 8 16 --sca_iters 3 --sca_prior anchor --lam_feat 0.0 --up_after 1 --mask_smooth l2 --smooth_w 0.001 --lam_temp 5.0 --tc_gap 2 --tc_start -1 --tc_anchor 1.0 --val_n 600 --workers 4 --ref_in 1 --ref_gap 30 --tex 1 --tex_flow 0.15 --epochs 100 --warmup 25 --ramp 12 --cls_max 1.0 --cls_cap 8.0 --adaptive 1 --anc_local 2 --tex_fixed 1 --enhance_only 1 --tgt_in 1 --tgt_frac 0.3 --tgt_pt 0.8 --lam_tgt_cls 1.0 --lam_tgt_mag 10 --out $OUT
