"""ConvNeXt-Tiny + ConvGRU temporal block trained with (windowed) TBPTT.

Hierarchical-Temporal-BP-Seg 의 BPTT 방식 차용:
  - spatial ConvGRU: hidden 이 feature map [B,C,7,7] (공간구조 보존, conv 게이트)
  - 비디오 프레임을 시퀀스로 흘리며 hidden 이월(causal)
  - TBPTT: 길이 seq_len 윈도우 안에서만 BPTT(gradient 절단), 윈도우마다 hidden reset
프레임별 3채널 [gray,CLAHE,scharr] → 공유 ConvNeXt → ConvGRU → 프레임별 분류(per-frame CE).
평가는 CM 스크립트에서 비디오 전체 스트리밍(hidden 이월)으로 수행.
Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python train_convnext_tbptt.py --seq 12 --epochs 40
"""
import argparse, os, glob, time, random
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from collections import Counter, defaultdict

BASE = '/home/doheon114/doheon114/meme_femoral/TinyUSFM/data/Cls'
GRAY, SCHARR = f'{BASE}/femoral', f'{BASE}/femoral_scharr'
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def parse(fn):
    vid, rest = fn.split('_f'); return vid, int(rest.split('.')[0])


class WinDS(Dataset):
    """비디오별 연속 seq_len 프레임 윈도우 + 프레임별 라벨(윈도우 내 TBPTT 단위)."""
    def __init__(self, split, seq=12, size=224, train=False, stride=None):
        self.size, self.train, self.seq = size, train, seq
        self.classes = sorted(c for c in os.listdir(f'{GRAY}/{split}/')
                              if os.path.isdir(f'{GRAY}/{split}/{c}'))
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        seqs = defaultdict(list)
        for c in self.classes:
            for p in glob.glob(f'{GRAY}/{split}/{c}/*.png'):
                vid, idx = parse(os.path.basename(p)); seqs[vid].append((idx, p, self.cls2idx[c]))
        stride = stride or (seq if not train else seq // 2)
        self.wins = []                                        # 각 윈도우: [(path,label)*seq]
        for vid, lst in seqs.items():
            lst.sort(); frames = [(p, y) for _, p, y in lst]; n = len(frames)
            if n < seq:
                self.wins.append(frames + [frames[-1]] * (seq - n)); continue
            for s in range(0, n - seq + 1, stride):
                self.wins.append(frames[s:s + seq])
            if (n - seq) % stride:
                self.wins.append(frames[n - seq:n])

    def __len__(self):
        return len(self.wins)

    def _one(self, gpath):
        rel = os.path.relpath(gpath, GRAY)
        g = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE); cl = _CLAHE.apply(g)
        sh = cv2.imread(os.path.join(SCHARR, rel), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(np.stack([g, cl, sh], -1), (self.size, self.size))

    def _augp(self):
        s = self.size; sc = random.uniform(0.7, 1.0); cs = int(s * sc)
        return dict(cs=cs, y0=random.randint(0, s - cs), x0=random.randint(0, s - cs),
                    flip=random.random() < 0.5, ang=random.uniform(-15, 15))

    def _apply(self, im, p):
        s = self.size
        im = cv2.resize(im[p['y0']:p['y0'] + p['cs'], p['x0']:p['x0'] + p['cs']], (s, s))
        if p['flip']: im = im[:, ::-1]
        M = cv2.getRotationMatrix2D((s / 2, s / 2), p['ang'], 1.0)
        return cv2.warpAffine(im, M, (s, s), borderMode=cv2.BORDER_REFLECT)

    def __getitem__(self, i):
        win = self.wins[i]; p = self._augp() if self.train else None
        imgs, ys = [], []
        for gp, y in win:
            im = self._one(gp)
            if self.train: im = self._apply(im, p)
            im = np.ascontiguousarray(im).astype(np.float32) / 255.0
            imgs.append((im - 0.5) / 0.5); ys.append(y)
        clip = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2))   # [S,3,H,W]
        return clip, torch.tensor(ys)


class ConvGRUCell(nn.Module):
    """Hierarchical-Temporal-BP-Seg 의 ConvGRUCell 차용(공간 순환)."""
    def __init__(self, in_dim, hid, k=3):
        super().__init__()
        pad = k // 2
        self.conv_zr = nn.Conv2d(in_dim + hid, 2 * hid, k, padding=pad)
        self.conv_h1 = nn.Conv2d(in_dim, hid, k, padding=pad)
        self.conv_h2 = nn.Conv2d(hid, hid, k, padding=pad)
        self.hid = hid

    def forward(self, x, h):
        z, r = torch.split(torch.sigmoid(self.conv_zr(torch.cat([x, h], 1))), self.hid, 1)
        h_ = torch.tanh(self.conv_h1(x) + r * self.conv_h2(h))
        return (1 - z) * h_ + z * h


class TBPTTNet(nn.Module):
    """backbone(공유 프레임 인코더) → 768ch feature map → ConvGRU → 프레임별 분류.
    backbone='convnext_tiny'(≈28M) 또는 'resnet50'(≈25M, 2048ch를 1x1로 768 투영)."""
    def __init__(self, n_cls, hid=256, drop_path=0.3, backbone='convnext_tiny'):
        super().__init__()
        self.backbone = backbone
        if backbone == 'convnext_tiny':
            base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                                        stochastic_depth_prob=drop_path)
            self.features = base.features                     # -> [.,768,7,7]
            self.proj = nn.Identity()
        elif backbone == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.features = nn.Sequential(*list(base.children())[:-2])  # -> [.,2048,7,7]
            self.proj = nn.Conv2d(2048, 768, 1)               # temporal block 동일화 위해 768로 투영
        else:
            raise NotImplementedError(backbone)
        self.tgru = ConvGRUCell(768, hid, 3)
        self.gn = nn.GroupNorm(32, hid)
        self.head = nn.Linear(hid, n_cls)
        self.hid = hid

    def encode(self, x):                                      # x [B,3,H,W] -> [B,768,7,7]
        return self.proj(self.features(x))

    def forward(self, x, h=None, ret_h=False):
        """x: [B,S,3,H,W]. 윈도우 내 hidden 스캔(BPTT). h 주면 이월(스트리밍 평가용)."""
        B, S, C, H, W = x.shape
        f = self.encode(x.reshape(B * S, C, H, W))            # [B*S,768,7,7]
        f = f.reshape(B, S, *f.shape[1:])
        if h is None:
            h = x.new_zeros(B, self.hid, f.shape[-2], f.shape[-1])
        outs = []
        for t in range(S):
            h = self.tgru(f[:, t], h)
            g = F.adaptive_avg_pool2d(self.gn(h), 1).flatten(1)
            outs.append(self.head(g))                         # 프레임별 logits [B,n_cls]
        out = torch.stack(outs, 1)                            # [B,S,n_cls]
        return (out, h) if ret_h else out


TBPTTConvNeXt = TBPTTNet   # backward-compat alias (기존 convnext 체크포인트 평가용)


@torch.no_grad()
def evaluate(model, loader, device, n_cls):
    """윈도우 단위 평가(프레임별). 스트리밍 CM 은 별도 스크립트."""
    model.eval(); c = t = 0; pc = [0] * n_cls; pt = [0] * n_cls
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)                                        # [B,S,n_cls]
        pred = out.argmax(-1).reshape(-1); y = y.reshape(-1)
        c += (pred == y).sum().item(); t += y.numel()
        for k in range(n_cls):
            m = y == k; pt[k] += m.sum().item(); pc[k] += (pred[m] == k).sum().item()
    return c / t, sum(pc[k] / pt[k] for k in range(n_cls) if pt[k]) / n_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seq', type=int, default=12)             # TBPTT 윈도우 길이
    ap.add_argument('--batch', type=int, default=8)            # 윈도우 개수(각 seq 프레임)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--drop_path', type=float, default=0.3)
    ap.add_argument('--label_smooth', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.99)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--backbone', choices=['convnext_tiny', 'resnet50'], default='convnext_tiny')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f'{"convnext" if args.backbone == "convnext_tiny" else "resnet50"}_tbptt_best.pt'
    device = torch.device('cuda'); ngpu = torch.cuda.device_count()

    ds = {s: WinDS(s, args.seq, args.size, train=(s == 'train')) for s in ('train', 'val', 'test')}
    dl = {s: DataLoader(ds[s], batch_size=args.batch, shuffle=(s == 'train'),
                        num_workers=args.workers, pin_memory=True, drop_last=(s == 'train'),
                        persistent_workers=True) for s in ds}
    classes = ds['train'].classes; n_cls = len(classes)
    print(f'classes={classes} | wins train={len(ds["train"])} val={len(ds["val"])} test={len(ds["test"])} '
          f'| seq={args.seq} gpus={ngpu}', flush=True)

    # class weights from per-frame labels in train windows
    cw = Counter(int(y) for w in ds['train'].wins for _, y in w)
    w = torch.tensor([sum(cw.values()) / (n_cls * cw[k]) for k in range(n_cls)], device=device)

    net = TBPTTNet(n_cls, 256, args.drop_path, backbone=args.backbone).to(device)
    model = nn.DataParallel(net) if ngpu > 1 else net
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    best = 0.0; stale = 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); rl = c = t = 0
        for x, y in dl['train']:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)                                # [B,S,n_cls]
                loss = crit(out.reshape(-1, n_cls), y.reshape(-1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v.detach().float(), alpha=1 - args.ema)
                    else:
                        ema[k] = v.detach().clone().float()
            pred = out.argmax(-1).reshape(-1); yy = y.reshape(-1)
            rl += loss.item() * yy.numel(); c += (pred == yy).sum().item(); t += yy.numel()
        sched.step()
        raw = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict({k: ema[k].to(v.dtype) for k, v in net.state_dict().items()})
        va, vb = evaluate(model, dl['val'], device, n_cls)
        te, tb = evaluate(model, dl['test'], device, n_cls)
        net.load_state_dict(raw)
        tag = ''
        if vb > best:
            best = vb; stale = 0
            torch.save({'model': {k: ema[k] for k in ema}, 'classes': classes, 'epoch': ep,
                        'seq': args.seq, 'hid': 256, 'backbone': args.backbone}, args.out)
            tag = '  *best'
        else:
            stale += 1
        print(f'[ep {ep:02d}/{args.epochs}] loss {rl/t:.4f} tr_acc {c/t:.4f} | '
              f'val_acc {va:.4f} val_bal {vb:.4f} | test_acc {te:.4f} test_bal {tb:.4f} | '
              f'{time.time()-t0:.0f}s{tag}', flush=True)
        if stale >= args.patience:
            print(f'early stop @ep{ep} (best val_bal={best:.4f})', flush=True); break
    print(f'done. best val_bal={best:.4f} -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
