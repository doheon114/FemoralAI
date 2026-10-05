"""공통 설정 / 전처리.

분류기(train_convnext_tbptt.py) 입력 규약을 그대로 재현한다:
  원본 해상도 gray -> [gray, CLAHE(gray), scharr(gray)] -> 224 resize -> (x-0.5)/0.5  (범위 [-1,1])
생성기 출력은 gray 채널을 기준으로 CLAHE/scharr 를 "다시 유도"(rederive)하여 분류기에 넣는다.
(3채널을 독립적으로 흔들면 scharr/CLAHE 채널만 조작하는 적대적 지름길이 생기기 때문)
"""
import os
import glob
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get('ANSIO_DATA_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 데이터 루트 ({split}/{CLASS}/*.png)
GRAY_DIR = ROOT                                                        # {split}/{CLASS}/*.png
SCHARR_DIR = os.path.join(ROOT, 'femoral_scharr')
CLS_CKPT = os.environ.get('ANSIO_CLS_CKPT') or os.path.join(ROOT, 'convnext_tbptt_best.pt')   # 환경변수로 분류기 교체

# 명세의 의미적 클래스 -> 데이터셋 폴더명 (체크포인트 'classes' 로 인덱스를 찾는다)
SEM2NAME = {'visible': 'NERVE_VISIBLE', 'anisotropic': 'ANISOTROPIC', 'absent': 'NERVE_NOT_VISIBLE'}

SIZE = int(os.environ.get('ANSIO_SIZE', 224))           # 환경변수로 해상도 교체 (32 의 배수)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def parse(fn):
    vid, rest = os.path.basename(fn).split('_f')
    return vid, int(rest.split('.')[0])


def list_frames(split, cls_name):
    if '+' in split:                                       # 'train+val': 여러 split 합침
        return sorted(q for sp in split.split('+') for q in list_frames(sp, cls_name))
    return sorted(glob.glob(os.path.join(GRAY_DIR, split, cls_name, '*.png')))


def videos(split, cls_names):
    """{vid: [(frame_idx, path, cls_name), ...]} (시간순)"""
    out = defaultdict(list)
    for c in cls_names:
        for p in list_frames(split, c):
            vid, idx = parse(p)
            out[vid].append((idx, p, c))
    for v in out.values():
        v.sort()
    return out


def load_input(gpath, size=SIZE):
    """분류기 평가 시와 동일한 3채널 입력(uint8 HxWx3). scharr 는 사전 계산 파일 사용."""
    rel = os.path.relpath(gpath, GRAY_DIR)
    g = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
    sp = os.path.join(SCHARR_DIR, rel)
    sh = cv2.imread(sp, cv2.IMREAD_GRAYSCALE) if os.path.exists(sp) else scharr_np(g)
    im = np.stack([g, _CLAHE.apply(g), sh], -1)
    return cv2.resize(im, (size, size)), g.shape[:2]


def tissue_mask(g_u8, size=SIZE, feather=4):
    """letterbox(검정 띠) 제외 조직 영역 prior mask [0,1] (분류기 학습 스크립트와 동일 규칙)."""
    f = g_u8.astype(np.float32) / 255.0
    rows, cols = f.mean(1) > 0.02, f.mean(0) > 0.02
    m = cv2.GaussianBlur((rows[:, None] & cols[None, :]).astype(np.float32), (0, 0), feather)
    return cv2.resize(m, (size, size))


_DIS = None


def ref_flow(x, r):
    """고정 정합 flow (학습 없음): DIS optical flow 로 현재 프레임 x 의 각 픽셀이 참조 r 의 어디에 해당하는지.
    x, r: [C,H,W] in [-1,1] (채널 0 = gray). -> [2,H,W] (dx/W, dy/H)  ->  warp(r)(p) = r(p + flow) ≈ x(p)."""
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    a = ((x[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    b = ((r[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    f = _DIS.calc(a, b, None)                               # a(p) ≈ b(p + f)
    H, W = a.shape
    return torch.from_numpy(np.stack([f[..., 0] / W, f[..., 1] / H])).float()


def to_tensor(im_u8):
    """HxWxC uint8 -> CxHxW float [-1,1]"""
    x = torch.from_numpy(np.ascontiguousarray(im_u8)).float().div(255.0)
    x = x.permute(2, 0, 1) if x.ndim == 3 else x[None]
    return x * 2 - 1


def scharr_np(g):
    b = cv2.GaussianBlur(g, (5, 5), 0)
    mag = np.hypot(cv2.Scharr(b, cv2.CV_32F, 1, 0), cv2.Scharr(b, cv2.CV_32F, 0, 1))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def original_hw():
    p = list_frames('train', SEM2NAME['visible'])[0]
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).shape[:2]


# ---------------------------------------------------------------- differentiable rederive
_G5 = torch.tensor([1., 4., 6., 4., 1.]) / 16.            # cv2.GaussianBlur((5,5), sigma=0) 고정 커널
_SX = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])


def scharr_torch(g01):
    """g01 [B,1,H,W] in [0,1] -> scharr magnitude, 이미지별 min-max 정규화 [0,1]. (미분 가능)"""
    dev, dt = g01.device, g01.dtype
    x = g01 * 255.0
    k = _G5.to(dev, dt)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode='reflect'), k.view(1, 1, 1, 5))   # reflect == BORDER_REFLECT_101
    x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode='reflect'), k.view(1, 1, 5, 1))
    xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
    sx = _SX.to(dev, dt)
    gx = F.conv2d(xp, sx.view(1, 1, 3, 3))
    gy = F.conv2d(xp, sx.t().contiguous().view(1, 1, 3, 3))
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mn = mag.amin(dim=(2, 3), keepdim=True)
    mx = mag.amax(dim=(2, 3), keepdim=True)
    return (mag - mn) / (mx - mn + 1e-6)


def clahe_st(g01_hi, out_hw):
    """CLAHE 는 cv2 로 계산하고, 역전파는 straight-through(identity) 로 근사."""
    with torch.no_grad():
        u8 = (g01_hi.float().clamp(0, 1) * 255).round().to(torch.uint8).squeeze(1).cpu().numpy()
        cl = np.stack([cv2.resize(_CLAHE.apply(a), (out_hw[1], out_hw[0])) for a in u8])
        cl = torch.from_numpy(cl).to(g01_hi.device).float().div(255.0)[:, None]
    return cl


def rederive(x, orig_hw=None):
    """x [B,3,h,w] in [-1,1] (채널0=gray 만 사용) -> 분류기 입력 [B,3,h,w] in [-1,1].
    원본 해상도(orig_hw)로 올려서 CLAHE/scharr 를 계산한 뒤 다시 h,w 로 내린다(데이터 파이프라인과 동일).
    bf16 autocast 는 0~255 스케일 scharr 정밀도를 떨어뜨리므로 fp32 로 강제한다."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _rederive(x, orig_hw)


def _rederive(x, orig_hw):
    g = (x[:, :1].float() + 1) / 2                         # [B,1,h,w] in [0,1]
    h, w = g.shape[-2:]
    hi = F.interpolate(g, size=orig_hw, mode='bilinear', align_corners=False) if orig_hw and tuple(orig_hw) != (h, w) else g
    sh = scharr_torch(hi)
    if sh.shape[-2:] != (h, w):
        sh = F.interpolate(sh, size=(h, w), mode='bilinear', align_corners=False)
    cl = clahe_st(hi, (h, w))
    cl = cl + (g - g.detach())                             # straight-through gradient
    return torch.cat([g, cl, sh], 1) * 2 - 1
