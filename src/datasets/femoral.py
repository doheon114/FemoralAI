"""Femoral nerve ultrasound frames as windows of consecutive frames (matches the classifier's WinDS).

Layout: `<root>/<split>/<CLASS>/<vid>_f<idx>.png` (448x448 grayscale), classes sorted alphabetically:
    0 = ANISOTROPIC, 1 = NERVE_NOT_VISIBLE, 2 = NERVE_VISIBLE
Frames of one video are spread over the class folders and re-assembled by file name; splits are video-level.

Each item is one window of `seq` consecutive frames:
    image  : float [S,1,H,W] grayscale in [-1,1]  (the generator operates on gray only)
    image3 : float [S,3,H,W] exact classifier input [gray, CLAHE, scharr] (optional, `return_exact=True`)
    label  : long  [S] per-frame GT class
    vid, start : window identification for visualization
    ctx, ctx_valid : (only if `context=P > 0`) the P frames preceding `start` [P,1,H,W] and a [P] validity mask
                     (left-padded with the first frame of the video where fewer than P exist) -> past-frame memory
"""
import glob
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ['ANISOTROPIC', 'NERVE_NOT_VISIBLE', 'NERVE_VISIBLE']
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def parse_name(fn):
    vid, rest = fn.split('_f')
    return vid, int(rest.split('.')[0])


class FemoralWindowDataset(Dataset):
    def __init__(self, root_dir, split, seq=12, img_size=224, stride=None, hflip=False,
                 scharr_dir=None, return_exact=False, only_windows_with=None, limit=None, context=0):
        self.root, self.split, self.seq, self.size = root_dir, split, seq, img_size
        self.hflip, self.return_exact, self.context = hflip, return_exact, context
        self.scharr_dir = scharr_dir or os.path.join(os.path.dirname(root_dir.rstrip('/')), 'femoral_scharr')
        classes = sorted(c for c in os.listdir(f'{root_dir}/{split}') if os.path.isdir(f'{root_dir}/{split}/{c}'))
        assert classes == CLASSES, classes
        self.cls2idx = {c: i for i, c in enumerate(classes)}

        seqs = defaultdict(list)
        for c in classes:
            for p in glob.glob(f'{root_dir}/{split}/{c}/*.png'):
                vid, idx = parse_name(os.path.basename(p))
                seqs[vid].append((idx, p, self.cls2idx[c]))
        stride = stride or seq
        self.wins = []  # (vid, start, [(path, label)] * seq)
        self.frames = {}  # vid -> full ordered [(path, label)] (for the past-frame context)
        for vid, lst in sorted(seqs.items()):
            lst.sort()
            frames = [(p, y) for _, p, y in lst]
            self.frames[vid] = frames
            n = len(frames)
            if n < seq:
                self.wins.append((vid, 0, frames + [frames[-1]] * (seq - n)))
                continue
            for s in range(0, n - seq + 1, stride):
                self.wins.append((vid, s, frames[s:s + seq]))
            if (n - seq) % stride:
                self.wins.append((vid, n - seq, frames[n - seq:n]))
        if only_windows_with is not None:  # keep windows containing at least one frame of this class
            self.wins = [w for w in self.wins if any(y == only_windows_with for _, y in w[2])]
        if limit:
            self.wins = self.wins[:limit]
        self.n_videos = len(seqs)

    def __len__(self):
        return len(self.wins)

    def _load(self, gpath):
        g = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
        if not self.return_exact:
            return cv2.resize(g, (self.size, self.size))[..., None]
        rel = os.path.relpath(gpath, self.root)
        cl = _CLAHE.apply(g)
        sh = cv2.imread(os.path.join(self.scharr_dir, rel), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(np.stack([g, cl, sh], -1), (self.size, self.size))

    def __getitem__(self, i):
        vid, start, win = self.wins[i]
        flip = self.hflip and random.random() < 0.5

        def load_clip(paths):
            imgs = []
            for gp in paths:
                im = self._load(gp)
                if flip:
                    im = im[:, ::-1]
                imgs.append((np.ascontiguousarray(im).astype(np.float32) / 255.0 - 0.5) / 0.5)
            return torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2))  # [T,C,H,W]

        clip = load_clip([gp for gp, _ in win])  # [S,C,H,W]
        out = {'label': torch.tensor([y for _, y in win]), 'vid': vid, 'start': start}
        if self.context > 0:
            P = self.context
            past = self.frames[vid][max(start - P, 0):start]
            n_pad = P - len(past)
            valid = torch.tensor([False] * n_pad + [True] * len(past))
            past = [self.frames[vid][0]] * n_pad + past
            out['ctx'] = load_clip([gp for gp, _ in past])[:, :1]  # gray only
            out['ctx_valid'] = valid
        if self.return_exact:
            out['image3'] = clip
            out['image'] = clip[:, :1]
        else:
            out['image'] = clip
        return out

    def label_counts(self):
        cnt = np.zeros(len(CLASSES), dtype=int)
        for _, _, w in self.wins:
            for _, y in w:
                cnt[y] += 1
        return cnt
