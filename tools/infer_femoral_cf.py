"""Run a trained femoral ANI->VIS counterfactual model over whole videos and export comparison videos.

For every video of the split (non-overlapping 12-frame windows, classifier hidden reset per window like
training/eval), writes `<out>/<vid>.mp4` with panels [input | counterfactual | |diff| heatmap] and a text bar
GT / f(x) / f(x_cf), plus `<out>/<split>_frames.csv` with per-frame probabilities and the L1 edit size.

Usage:
  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 python tools/infer_femoral_cf.py \
      -r training_logs/femoral/<run_dir> [--ckpt best.pth] [--split val] [--vids 00437095 ...] [--all-frames]
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from src.datasets.femoral import CLASSES, FemoralWindowDataset
from src.models import build_model

SHORT = {'ANISOTROPIC': 'ANI', 'NERVE_NOT_VISIBLE': 'ABS', 'NERVE_VISIBLE': 'VIS'}


def to_u8(img):  # [-1,1] -> uint8
    return ((np.clip(img, -1, 1) + 1) * 127.5).round().astype(np.uint8)


def panel(x, x_cf, gt, fx, fcf, c, scale=2):
    a, b = to_u8(x), to_u8(x_cf)
    d = np.abs(b.astype(np.float32) - a.astype(np.float32)) / 255.0
    heat = cv2.applyColorMap((np.clip(d / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a, b = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    row = np.concatenate([a, b, heat], 1)
    if scale != 1:
        row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((28, row.shape[1], 3), np.uint8)
    col = (0, 200, 255) if c else (200, 200, 200)
    txt = f'GT {SHORT[CLASSES[gt]]} | f(x) {SHORT[CLASSES[int(fx.argmax())]]} {fx.max():.2f} | f(x_cf) {SHORT[CLASSES[int(fcf.argmax())]]} {fcf.max():.2f} pVIS {fcf[2]:.2f} | |d| {d.mean():.3f}' + ('  [inpaint]' if c else '')
    cv2.putText(bar, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return np.concatenate([bar, row], 0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--run_dir', required=True)
    ap.add_argument('--ckpt', default='best.pth')
    ap.add_argument('--split', default='val')
    ap.add_argument('--vids', nargs='*', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = torch.device(args.device)

    run = Path(args.run_dir)
    opt = edict(yaml.safe_load(open(run / 'hparams.yaml')))
    model = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size).to(dev).eval()
    state = torch.load(run / 'checkpoints' / args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    print(f'Loaded {args.ckpt} (epoch {state["epoch"]}, best_score {state.get("best_score")})')

    ds = FemoralWindowDataset(opt.dataset.root_dir, args.split, seq=opt.dataset.seq, img_size=opt.dataset.img_size[0], stride=opt.dataset.seq,
                              context=opt.dataset.get('context', 0))
    out = Path(args.out or run / f'inference_{args.split}_{args.ckpt.replace(".pth", "")}')
    out.mkdir(parents=True, exist_ok=True)

    per_vid = defaultdict(list)
    for i, (vid, start, _) in enumerate(ds.wins):
        if args.vids is None or vid in args.vids:
            per_vid[vid].append((start, i))

    rows = []
    for vid, wins in per_vid.items():
        wins.sort()
        writer, seen = None, set()
        for start, i in tqdm(wins, desc=vid, leave=False):
            item = ds[i]
            x = item['image'][None].to(dev)
            ctx = (item['ctx'][None].to(dev), item['ctx_valid'][None].to(dev)) if 'ctx' in item else (None, None)
            x_cf, p_x, p_cf, c = model.counterfactual(x, *ctx)
            x, x_cf = x[0, :, 0].cpu().numpy(), x_cf[0, :, 0].cpu().numpy()
            p_x, p_cf, c, y = p_x[0].cpu().numpy(), p_cf[0].cpu().numpy(), c[0].cpu().numpy(), item['label'].numpy()
            for t in range(x.shape[0]):
                fidx = start + t
                if fidx in seen:  # tail window overlaps the previous one
                    continue
                seen.add(fidx)
                fr = panel(x[t], x_cf[t], int(y[t]), p_x[t], p_cf[t], int(c[t]))
                if writer is None:
                    writer = cv2.VideoWriter(str(out / f'{vid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (fr.shape[1], fr.shape[0]))
                writer.write(fr)
                rows.append(dict(vid=vid, frame=fidx, gt=CLASSES[int(y[t])], cond=int(c[t]),
                                 p_ani=p_x[t][0], p_abs=p_x[t][1], p_vis=p_x[t][2],
                                 cf_p_ani=p_cf[t][0], cf_p_abs=p_cf[t][1], cf_p_vis=p_cf[t][2],
                                 l1=float(np.abs(x_cf[t] - x[t]).mean() / 2)))
        if writer is not None:
            writer.release()
    with open(out / f'{args.split}_frames.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    r = np.array([(x['cond'], x['gt'] == 'ANISOTROPIC', CLASSES[int(np.argmax([x['cf_p_ani'], x['cf_p_abs'], x['cf_p_vis']]))] == 'NERVE_VISIBLE',
                   CLASSES[int(np.argmax([x['p_ani'], x['p_abs'], x['p_vis']]))] == CLASSES[int(np.argmax([x['cf_p_ani'], x['cf_p_abs'], x['cf_p_vis']]))], x['l1']) for x in rows], dtype=float)
    inp = r[:, 0] == 1
    print(f'{len(rows)} frames | flip_pred {r[inp, 2].mean():.3f} (n={inp.sum()}) | flip_gt {r[r[:, 1] == 1, 2].mean():.3f} | '
          f'keep {r[~inp, 3].mean():.3f} | l1_ani {r[inp, 4].mean():.4f} l1_keep {r[~inp, 4].mean():.4f}')
    print('Wrote', out)


if __name__ == '__main__':
    main()
