"""Frozen femoral 3-class classifier (ConvNeXt-Tiny + ConvGRU, trained with TBPTT).

Architecture is a 1:1 copy of `TinyUSFM/data/Cls/femoral/train_convnext_tbptt.py::TBPTTNet` so the
checkpoint `convnext_tbptt_best.pt` loads strictly. Classes (sorted folder names):
    0 = ANISOTROPIC, 1 = NERVE_NOT_VISIBLE, 2 = NERVE_VISIBLE

Input per frame is 3 channels [gray, CLAHE(gray), scharr(gray)] at 224x224, normalized to [-1, 1].
`derive_channels` rebuilds those channels from a (generated) grayscale frame so the generator only has to
produce a single grayscale image and gradients still reach it through the classifier:
    * gray   -> identity
    * CLAHE  -> cv2 on CPU, straight-through estimator for the gradient
    * scharr -> differentiable conv re-implementation of the cv2 recipe (blur 5x5 -> Scharr magnitude -> min-max)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CLASSES = ['ANISOTROPIC', 'NERVE_NOT_VISIBLE', 'NERVE_VISIBLE']
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# resolution the classifier's channels were precomputed at (before the 224 resize)
NATIVE_RES = 448


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
    def __init__(self, n_cls, hid=256, drop_path=0.0, backbone='convnext_tiny'):
        super().__init__()
        self.backbone = backbone
        if backbone == 'convnext_tiny':
            base = models.convnext_tiny(weights=None, stochastic_depth_prob=drop_path)
            self.features = base.features
            self.proj = nn.Identity()
        elif backbone == 'resnet50':
            base = models.resnet50(weights=None)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.proj = nn.Conv2d(2048, 768, 1)
        else:
            raise NotImplementedError(backbone)
        self.tgru = ConvGRUCell(768, hid, 3)
        self.gn = nn.GroupNorm(32, hid)
        self.head = nn.Linear(hid, n_cls)
        self.hid = hid

    def encode(self, x):
        return self.proj(self.features(x))

    def forward(self, x, h=None, ret_h=False):
        """x: [B,S,3,H,W] -> per-frame logits [B,S,n_cls]; hidden is reset per window unless `h` is given."""
        B, S, C, H, W = x.shape
        f = self.encode(x.reshape(B * S, C, H, W))
        f = f.reshape(B, S, *f.shape[1:])
        if h is None:
            h = x.new_zeros(B, self.hid, f.shape[-2], f.shape[-1])
        outs = []
        for t in range(S):
            h = self.tgru(f[:, t], h)
            g = F.adaptive_avg_pool2d(self.gn(h), 1).flatten(1)
            outs.append(self.head(g))
        out = torch.stack(outs, 1)
        return (out, h) if ret_h else out


def load_tbptt_classifier(ckpt_path: str, device='cuda') -> TBPTTNet:
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    classes = state.get('classes', CLASSES)
    assert list(classes) == CLASSES, classes
    net = TBPTTNet(len(classes), hid=state.get('hid', 256), backbone=state.get('backbone', 'convnext_tiny'))
    net.load_state_dict(state['model'], strict=True)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    print(f'Restored TBPTT classifier from {ckpt_path} (epoch={state.get("epoch")}, seq={state.get("seq")})')
    return net


# --------------------------------------------------------------------------------------------------
# channel derivation
# --------------------------------------------------------------------------------------------------
def _gauss_kernel5(dtype, device):
    # cv2.GaussianBlur((5,5), sigma=0) -> sigma = 0.3*((5-1)*0.5-1)+0.8 = 1.1
    k1 = cv2.getGaussianKernel(5, 0).astype(np.float32).reshape(-1)
    k2 = np.outer(k1, k1)
    return torch.tensor(k2, dtype=dtype, device=device)[None, None]


_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)


def scharr_channel(gray01: torch.Tensor, native_res: int = NATIVE_RES) -> torch.Tensor:
    """gray01: [N,1,H,W] in [0,1] -> scharr channel in [-1,1] (differentiable).

    Mirrors the offline recipe (computed at `native_res`, then resized to H x W):
    GaussianBlur(5x5, sigma=0) -> Scharr magnitude -> per-image min-max to [0,255] -> uint8 -> resize.
    """
    N, _, H, W = gray01.shape
    x = gray01 * 255.0
    if native_res and native_res != H:
        x = F.interpolate(x, size=(native_res, native_res), mode='bilinear', align_corners=False)
    dt, dev = x.dtype, x.device
    x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), _gauss_kernel5(dt, dev))
    kx = torch.tensor(_SCHARR_X, dtype=dt, device=dev)[None, None]
    ky = kx.transpose(2, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
    gx, gy = F.conv2d(xp, kx), F.conv2d(xp, ky)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mn = mag.amin(dim=(2, 3), keepdim=True)
    mx = mag.amax(dim=(2, 3), keepdim=True)
    mag = (mag - mn) / (mx - mn + 1e-6)
    if native_res and native_res != H:
        mag = F.interpolate(mag, size=(H, W), mode="bilinear", align_corners=False)
    return mag.clamp(0, 1) * 2 - 1


def clahe_channel(gray01: torch.Tensor, native_res: int = NATIVE_RES) -> torch.Tensor:
    """gray01: [N,1,H,W] in [0,1] -> CLAHE channel in [-1,1]; cv2 on CPU with straight-through gradient."""
    N, _, H, W = gray01.shape
    g = gray01.detach()
    if native_res and native_res != H:
        g = F.interpolate(g, size=(native_res, native_res), mode='bilinear', align_corners=False)
    g_np = (g.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()[:, 0]
    out = np.empty((N, H, W), dtype=np.uint8)
    for i in range(N):
        cl = _CLAHE.apply(g_np[i])
        out[i] = cv2.resize(cl, (W, H)) if cl.shape != (H, W) else cl
    cl = torch.from_numpy(out).to(gray01.device, gray01.dtype).unsqueeze(1) / 255.0
    # straight-through: forward = CLAHE(gray), backward = identity w.r.t. gray
    cl = cl + (gray01 - gray01.detach())
    return cl * 2 - 1


def derive_channels(gray: torch.Tensor) -> torch.Tensor:
    """gray: [N,1,H,W] in [-1,1] -> [N,3,H,W] = [gray, CLAHE, scharr] in [-1,1]."""
    gray01 = (gray + 1) / 2
    return torch.cat([gray, clahe_channel(gray01), scharr_channel(gray01)], dim=1)
