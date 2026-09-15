"""Sanity check: the frozen TBPTT classifier reproduces its logged val/test accuracy inside this repo,
both with the exact cv2 3-channel input and with channels derived in-graph from the gray frame
(the path the generator output goes through).

Logged (train_convnext_tbptt.log, ep 9): val_acc 0.8394 val_bal 0.8128 | test_acc 0.8610 test_bal 0.8009
Usage: PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 python tools/verify_femoral_classifier.py
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.femoral import FemoralWindowDataset
from src.models.tbptt_classifier import derive_channels, load_tbptt_classifier

ROOT = '/home/doheon114/doheon114/meme_femoral/TinyUSFM/data/Cls/femoral'
CKPT = f'{ROOT}/convnext_tbptt_best.pt'


@torch.no_grad()
def run(net, loader, derived):
    c = t = 0
    pc, pt = np.zeros(3), np.zeros(3)
    for b in tqdm(loader, leave=False):
        y = b['label'].cuda().reshape(-1)
        if derived:
            x = b['image'].cuda()
            B, S = x.shape[:2]
            x = derive_channels(x.reshape(B * S, *x.shape[2:])).reshape(B, S, 3, *x.shape[3:])
        else:
            x = b['image3'].cuda()
        pred = net(x).argmax(-1).reshape(-1)
        c += (pred == y).sum().item()
        t += y.numel()
        for k in range(3):
            m = y == k
            pt[k] += m.sum().item()
            pc[k] += (pred[m] == k).sum().item()
    return c / t, float(np.mean(pc / np.maximum(pt, 1)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--ckpt', default=CKPT)
    args = ap.parse_args()
    net = load_tbptt_classifier(args.ckpt)
    for split in ('val', 'test'):
        ds = FemoralWindowDataset(args.root, split, seq=12, return_exact=True)
        dl = DataLoader(ds, batch_size=8, num_workers=8)
        for derived in (False, True):
            acc, bal = run(net, dl, derived)
            print(f'{split:5s} derived={derived!s:5s} acc={acc:.4f} bal={bal:.4f}', flush=True)
