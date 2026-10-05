"""Frozen 3-class classifier (ConvNeXt-Tiny + ConvGRU, TBPTT 학습본) 래퍼.

- 체크포인트 'classes' = sorted 폴더명 = ['ANISOTROPIC','NERVE_NOT_VISIBLE','NERVE_VISIBLE']
  -> 명세의 (Visible=0, Anisotropic=1, Absent=2) 와 인덱스가 다르므로 이름으로 매핑한다.
- 원 모델은 비디오 스트리밍(hidden 이월) 모델이다. 단일 프레임 판정은 같은 프레임 feature 로
  GRU 를 `repeat` 번 돌린 정적 클립(static clip)의 마지막 logit 을 사용한다(encoder 는 1회만 실행).

사전 점검:  python classifier.py --split val   (repeat / rederive 설정별 정확도·확률 확인)
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from common import CLS_CKPT, SEM2NAME, list_frames, load_input, to_tensor, rederive, original_hw


class ConvGRUCell(nn.Module):
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
    """train_convnext_tbptt.TBPTTNet 과 동일한 state_dict 키 구조(가중치 다운로드 없이 생성)."""
    def __init__(self, n_cls, hid=256, backbone='convnext_tiny', in_proj=False, skip=False, temporal=True):
        super().__init__()
        self.temporal = temporal                           # False: 프레임 모델(--no_temporal), GAP -> fhead(LN+Linear)
        if backbone == 'convnext_tiny':
            self.features = models.convnext_tiny(weights=None, stochastic_depth_prob=0.0).features
            self.proj = nn.Identity()
        elif backbone == 'convnext_tiny_384':               # timm fb_in22k_ft_in1k_384 기반 (가중치는 체크포인트에서 로드)
            import timm
            self.features = timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384', pretrained=False,
                                              num_classes=0, global_pool='')
            self.proj = nn.Identity()
        elif backbone == 'resnet50':
            self.features = nn.Sequential(*list(models.resnet50(weights=None).children())[:-2])
            self.proj = nn.Conv2d(2048, 768, 1)
        else:
            raise NotImplementedError(backbone)
        # in_proj: 768->hid 1x1+GN 후 GRU / skip: head 에 현재 프레임 feature GAP concat (train_convnext_tbptt --in_proj --skip)
        self.in_proj = nn.Sequential(nn.Conv2d(768, hid, 1), nn.GroupNorm(32, hid)) if in_proj else None
        assert in_proj or not skip, 'skip 은 in_proj 필요'
        self.skip = skip
        self.tgru = ConvGRUCell(hid if in_proj else 768, hid, 3)
        self.gn = nn.GroupNorm(32, hid)
        self.head = nn.Linear(hid * (2 if skip else 1), n_cls)
        self.hid = hid
        if not temporal:
            self.fhead = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, n_cls))

    def encode(self, x):
        f = self.proj(self.features(x))
        return self.in_proj(f) if self.in_proj is not None else f

    def step(self, f, h):
        h = self.tgru(f, h)
        g = F.adaptive_avg_pool2d(self.gn(h), 1).flatten(1)
        if self.skip:
            g = torch.cat([g, F.adaptive_avg_pool2d(f, 1).flatten(1)], 1)
        return self.head(g), h


class TimmGRUNet(nn.Module):
    """godok/train_convnext_tbptt.TemporalNet 호환: timm ConvNeXt pooled feature + 벡터 GRU,
    head(cat[f, o]). 입력은 raw gray 1채널 -> 3채널 복제 -> ImageNet 정규화 (CLAHE/scharr 없음)."""
    def __init__(self, sd, model='convnext_base.fb_in22k_ft_in1k'):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model, pretrained=False, num_classes=0)
        self.hid = sd['gru.weight_hh_l0'].shape[1]
        self.gru = nn.GRU(self.backbone.num_features, self.hid, batch_first=True)
        self.head = nn.Linear(self.backbone.num_features + self.hid, sd['head.weight'].shape[0])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def logits(self, x, repeat):
        g = (x[:, :1] + 1) / 2                             # 채널0=gray (direct/rederive 모두 동일)
        f = self.backbone((g.expand(-1, 3, -1, -1) - self.mean) / self.std)
        with torch.backends.cudnn.flags(enabled=False):     # cudnn RNN 은 eval 모드 backward 불가 (입력 grad 필요)
            o, _ = self.gru(f[:, None].expand(-1, repeat, -1))  # 정적 클립: 같은 feature 를 repeat 프레임
        return self.head(torch.cat([f, o[:, -1]], -1))


class FrozenClassifier(nn.Module):
    def __init__(self, ckpt_path=CLS_CKPT, repeat=4):
        super().__init__()
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model' not in ck and 'gru.weight_ih_l0' in ck:  # godok convnext_base_tbptt (raw state_dict)
            import os
            self.classes = ['ANISOTROPIC', 'NERVE_NOT_VISIBLE', 'NERVE_VISIBLE']
            self.net = TimmGRUNet(ck, os.environ.get('ANSIO_CLS_TIMM', 'convnext_base.fb_in22k_ft_in1k'))
            self.net.load_state_dict(ck, strict=True)
            self.timm_gru = True
            self.in_size = int(os.environ.get('ANSIO_CLS_SIZE') or 384)
        else:
            self.timm_gru = False
            self._init_tbptt(ck)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        idx = {n: i for i, n in enumerate(self.classes)}
        self.i_vis = idx[SEM2NAME['visible']]
        self.i_ani = idx[SEM2NAME['anisotropic']]
        self.i_abs = idx[SEM2NAME['absent']]
        self.repeat = repeat

    def _init_tbptt(self, ck):
        self.classes = list(ck['classes'])
        for k, d in (('gru_k', 3), ('gru_dil', 1), ('gru_grid', 0), ('gru_type', 'std'), ('pool', 'gap'), ('ms', False),
                     ('ordinal', False), ('unet', 0), ('tsm', 0), ('temporal', 'convgru')):
            assert ck.get(k, d) == d, f'분류기 래퍼 미지원 옵션 {k}={ck.get(k)}'
        self.net = TBPTTNet(len(self.classes), ck.get('hid', 256), ck.get('backbone', 'convnext_tiny'),
                            in_proj=ck.get('in_proj', False), skip=ck.get('skip', False),
                            temporal='fhead.0.weight' not in ck['model'])   # Cls --no_temporal 프레임 모델 감지
        sd = self.net.state_dict()
        self.net.load_state_dict({k: ck['model'][k].to(v.dtype) for k, v in sd.items()}, strict=True)
        # 분류기 학습 해상도. 생성기 해상도(SIZE)와 다르면 forward 에서 antialias 다운샘플 (예: 384 편집 -> 224 판정)
        import os
        self.in_size = int(os.environ.get('ANSIO_CLS_SIZE') or (384 if ck.get('backbone') == 'convnext_tiny_384' else 224))

    def train(self, mode=True):                            # 항상 eval 모드 유지
        return super().train(False)

    def forward(self, x, repeat=None, eot=False):
        """x [B,3,H,W] in [-1,1] -> 3-class logits (체크포인트 인덱스 순서). H != in_size 면 antialias 리사이즈."""
        if x.shape[-1] != self.in_size or x.shape[-2] != self.in_size:
            x = F.interpolate(x.float(), size=(self.in_size, self.in_size), mode='bilinear', antialias=True, align_corners=False)
        if eot:                                            # 가벼운 EOT: 랜덤 flip + 평행이동(분류기 학습 증강 범위 내)
            if random.random() < 0.5:
                x = x.flip(-1)
            dy, dx = random.randint(-8, 8), random.randint(-8, 8)
            x = F.pad(x, (8, 8, 8, 8), mode='reflect')[..., 8 + dy:8 + dy + x.shape[-2], 8 + dx:8 + dx + x.shape[-1]]
        if self.timm_gru:
            return self.net.logits(x, repeat or self.repeat)
        f = self.net.encode(x)
        if not self.net.temporal:                          # 프레임 모델: repeat 무관
            return self.net.fhead(F.adaptive_avg_pool2d(f, 1).flatten(1))
        h = f.new_zeros(f.shape[0], self.net.hid, f.shape[-2], f.shape[-1])
        for _ in range(repeat or self.repeat):
            logits, h = self.net.step(f, h)
        return logits

    def sem_probs(self, logits):
        """3-class softmax 확률을 명세 순서(visible, anisotropic, absent)로."""
        p = logits.float().softmax(-1)
        return p[:, self.i_vis], p[:, self.i_ani], p[:, self.i_abs]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=300, help='클래스별 샘플 프레임 수')
    ap.add_argument('--repeats', type=int, nargs='+', default=[1, 4, 12])
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = FrozenClassifier().to(dev)
    ohw = original_hw()
    print(f'classes={C.classes}  vis={C.i_vis} ani={C.i_ani} abs={C.i_abs}  orig_hw={ohw}')
    random.seed(0)
    for sem in ('visible', 'anisotropic', 'absent'):
        paths = list_frames(args.split, SEM2NAME[sem])
        paths = random.sample(paths, min(args.n, len(paths)))
        x = torch.stack([to_tensor(load_input(p)[0]) for p in paths]).to(dev)
        tgt = {'visible': C.i_vis, 'anisotropic': C.i_ani, 'absent': C.i_abs}[sem]
        for mode in ('direct', 'rederive'):
            xi = x if mode == 'direct' else rederive(x, ohw)
            for r in args.repeats:
                lg = torch.cat([C(xi[i:i + 64], repeat=r) for i in range(0, len(xi), 64)])
                pv, pa, pb = C.sem_probs(lg)
                acc = (lg.argmax(-1) == tgt).float().mean().item()
                print(f'[{sem:11s}] {mode:8s} repeat={r:2d} acc={acc:.3f} | '
                      f'P(vis)={pv.mean():.3f} P(ani)={pa.mean():.3f} P(abs)={pb.mean():.3f}')


if __name__ == '__main__':
    main()
