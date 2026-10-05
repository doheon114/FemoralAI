"""Loss 모음.

L_total(G) = λ_cls L_cls + λ_gan L_gan + λ_rec L_rec + λ_tv L_tv  (+ λ_area L_area : 학습형 위치 마스크 정규화)

L_cls (targeted, 명세):  -log( e^{z_ani} / (e^{z_vis} + e^{z_ani}) ) = softplus(z_vis - z_ani)   (Absent logit 무시)
  + 선택적 보조항 β·(-log softmax3(z)_ani): 목표가 3-class 확률(P_ani>=0.85)이므로 Absent 로 새는 것 방지.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ classifier loss
def targeted_cls_loss(logits, i_vis, i_ani, beta3=0.0):
    """i_vis = source 클래스 인덱스, i_ani = target 클래스 인덱스 (ani2vis 에서는 서로 바뀌어 전달됨)."""
    z = logits.float()
    l = F.softplus(z[:, i_vis] - z[:, i_ani])               # 2-way CE toward anisotropic
    if beta3 > 0:
        l = l + beta3 * (-F.log_softmax(z, -1)[:, i_ani])
    return l.mean()


def hinge_cls_loss(logits, i_src, i_tgt, pt=0.7, ps=0.15, i_abs=None, pb=0.0):
    """성공 기준형 hinge (3-class softmax, Absent 포함): P(tgt)>=pt & P(src)<=ps 를 넘으면 gradient 0.
    이미 성공한 샘플을 더 밀지 않아 분류기 과공략(adversarial) 과 불필요한 수정을 줄인다.
    pb > 0: P(abs)<=pb 항 추가 (target 대신 Absent 로 새는 것 직접 벌점)."""
    lp = F.log_softmax(logits.float(), -1)
    l = F.relu(math.log(pt) - lp[:, i_tgt]) + F.relu(lp[:, i_src] - math.log(ps))
    if pb > 0 and i_abs is not None:
        l = l + F.relu(lp[:, i_abs] - math.log(pb))
    return l.mean()


def patch_weight(m_eff, d_out):
    """m_eff [B,1,H,W] -> PatchGAN 출력 격자 가중치 [B,1,h',w'] (patch 가 덮는 수정량 평균, detach)."""
    return F.adaptive_avg_pool2d(m_eff.detach().float(), d_out.shape[-2:])


# ------------------------------------------------------------------ adversarial
class GANLoss(nn.Module):
    def __init__(self, mode='lsgan'):
        super().__init__()
        assert mode in ('lsgan', 'bce')
        self.mode = mode

    def _l(self, pred, real, w=None):
        """w: patch 가중치 [B,1,h',w'] (None = 전 patch 균등). masked PatchGAN: 수정된 patch 만 판정."""
        pred = pred.float()
        t = torch.ones_like(pred) if real else torch.zeros_like(pred)
        if self.mode == 'lsgan':
            l = F.mse_loss(pred, t, reduction='none')
        else:
            l = F.binary_cross_entropy_with_logits(pred, t, reduction='none')
        return l.mean() if w is None else (l * w).sum() / (w.sum() + 1e-6)

    def d_loss(self, d_real, d_fake, w_fake=None):
        return 0.5 * (self._l(d_real, True) + self._l(d_fake, False, w_fake))

    def g_loss(self, d_fake, w=None):
        return self._l(d_fake, True, w)


# ------------------------------------------------------------------ reconstruction
def _gauss(ws=11, sigma=1.5, device='cpu'):
    g = torch.exp(-(torch.arange(ws, device=device).float() - ws // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y, data_range=2.0):
    """x,y [B,C,H,W] -> SSIM map (채널별 depthwise)."""
    C = x.shape[1]
    w = _gauss(device=x.device).expand(C, 1, -1, -1)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, padding=5, groups=C)
    mu_y = F.conv2d(y, w, padding=5, groups=C)
    sxx = F.conv2d(x * x, w, padding=5, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=C) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def rec_loss(out, ref, has_ref):
    """L1 + (1-SSIM), pseudo-pair 가 있는 샘플에만. out/ref [B,C,H,W] in [-1,1]."""
    w = has_ref.float().view(-1)
    if w.sum() == 0:
        return out.new_zeros(())
    out, ref = out.float(), ref.float()
    l1 = (out - ref).abs().mean((1, 2, 3))
    ss = 1 - ssim_map(out, ref).mean((1, 2, 3))
    return ((l1 + ss) * w).sum() / w.sum()


def masked_rec_loss(out, ref, has_ref, wmap):
    """수정 영역(wmap=m_eff detach)에서만 pseudo-pair 참조와 비교. ref 는 인접 다른 프레임이라
    speckle 이 어긋나므로 L1 은 저주파(avg 5x5) 로, 구조는 SSIM 으로 본다. 마스크 밖(=입력 그대로)은 제외."""
    w = has_ref.float().view(-1)
    if w.sum() == 0:
        return out.new_zeros(())
    out, ref, wm = out.float(), ref.float(), wmap.detach().float()
    lp = lambda t: F.avg_pool2d(t, 5, 1, 2, count_include_pad=False)
    lmap = (lp(out) - lp(ref)).abs().mean(1, keepdim=True) + (1 - ssim_map(out, ref)).mean(1, keepdim=True)
    l = (lmap * wm).sum((1, 2, 3)) / (wm.sum((1, 2, 3)) + 1e-6)
    return (l * w).sum() / w.sum()


def mag_loss(out, x):
    """최소 수정: 이미지 전체 평균 |I_out - I_in| (마스크 밖은 0) -> 면적 x 세기 를 함께 벌한다."""
    return (out.float() - x.float()).abs().mean()


# ------------------------------------------------------------------ regularizers
def masked_tv(delta, m_eff):
    """마스크 내부에 실제로 주입된 residual (ΔI ⊙ M_eff) 의 anisotropic TV (L1)."""
    r = (delta * m_eff).float()
    tv = (r[..., 1:, :] - r[..., :-1, :]).abs().mean() + (r[..., :, 1:] - r[..., :, :-1]).abs().mean()
    return tv


def seam_loss(delta, m_eff):
    """마스크 경계 띠(4 m(1-m), detach)에서 주입 residual (ΔI ⊙ M_eff) 의 gradient L1.
    내부 질감은 자유롭게 두고, 편집 영역 가장자리에 새 윤곽선(뚜렷한 경계)이 생기는 것만 벌한다."""
    m = m_eff.float()
    r = delta[:, :1].float() * m
    b = (4 * m * (1 - m)).detach()
    gy, gx = (r[..., 1:, :] - r[..., :-1, :]).abs(), (r[..., :, 1:] - r[..., :, :-1]).abs()
    by, bx = torch.maximum(b[..., 1:, :], b[..., :-1, :]), torch.maximum(b[..., :, 1:], b[..., :, :-1])
    return ((gy * by).sum() + (gx * bx).sum()) / (by.sum() + bx.sum() + 1e-6)


def mask_area_loss(m_hat, m_prior, target=0.0):
    """위치 마스크 면적(prior 대비 비율). target 초과분만 페널티 -> 국소적 변화 유도."""
    a = (m_hat * m_prior).float().sum((1, 2, 3)) / (m_prior.float().sum((1, 2, 3)) + 1e-6)
    return F.relu(a - target).mean()


def mask_compact_loss(m_eff, eps=1e-6):
    """Scale-invariant compactness: 2π·Var(p)/Area - 1  (>= 0; 원판 0, 단일 가우시안 1, 2~4 blob ≈ 11~12, 옅은 haze ≈ 9).
    p = [0,1]^2 정규화 좌표, 마스크 값을 질량으로 보는 공간 2차 모멘트.
    여러 blob 으로 분산되거나 옅게 넓게 퍼질수록(질량 대비 분산 큼) 커진다. 크기 자체는 벌하지 않음."""
    m = m_eff.float()[:, 0]                                 # [B,H,W]
    B, H, W = m.shape
    ys = (torch.arange(H, device=m.device).float() + 0.5) / H
    xs = (torch.arange(W, device=m.device).float() + 0.5) / W
    mass = m.sum((1, 2)) + eps
    cy = (m.sum(2) * ys).sum(1) / mass
    cx = (m.sum(1) * xs).sum(1) / mass
    var = ((m.sum(2) * (ys[None] - cy[:, None]) ** 2).sum(1) +
           (m.sum(1) * (xs[None] - cx[:, None]) ** 2).sum(1)) / mass
    area = mass / (H * W)                                   # 정규화 면적 (peak=1 기준)
    return F.relu(2 * torch.pi * var / (area + eps) - 1).mean()


def mask_grad_l2(m_hat):
    """L2 gradient (Dirichlet) 평활: 같은 높이의 경계라도 급할수록 벌점이 커서 완만한 전이를 선호.
    (L1 TV 는 경계 폭에 무관 -> 계단 경계를 막지 못함). 좌표를 [0,1] 로 정규화해 해상도 무관 스케일."""
    m = m_hat.float()
    H, W = m.shape[-2:]
    return (((m[..., 1:, :] - m[..., :-1, :]) * H) ** 2).mean() + (((m[..., :, 1:] - m[..., :, :-1]) * W) ** 2).mean()


def mask_tv(m_hat):
    m = m_hat.float()
    return (m[..., 1:, :] - m[..., :-1, :]).abs().mean() + (m[..., :, 1:] - m[..., :, :-1]).abs().mean()


def feat_completion_loss(sca_feats, tgt_feats, m_spec, has_ref):
    """SCA 로 completion 된 feature 가 pseudo-pair target 프레임(인접 Nerve Visible)의 plain feature 와
    target prior(M_spec) 영역 안에서 방향이 같아지도록 (1 - cos). 픽셀 inpainting 이 아니라 feature 감독."""
    w = has_ref.float().view(-1)
    if not sca_feats or w.sum() == 0:
        return m_spec.new_zeros(()) if m_spec is not None else torch.zeros(())
    ls = []
    for f, t in zip(sca_feats, tgt_feats):
        mm = F.interpolate(m_spec.float(), size=f.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        cos = F.cosine_similarity(f.float(), t.float(), dim=1)             # [B,h,w]
        ls.append(((1 - cos) * mm).sum((1, 2)) / (mm.sum((1, 2)) + 1e-6))
    l = torch.stack(ls, 0).mean(0)
    return (l * w).sum() / w.sum()


def speckle_loss(out, x, m_eff, k=7, eps=1e-4):
    """speckle 보존: 편집 영역(m_eff, detach) 안에서 출력/입력 gray 의 국소 Laplacian 에너지(k×k 평균) 로그비 |log E_out - log E_in|.
    blur(에너지 감소)와 과선명/고주파 적대 패턴(에너지 증가)을 대칭으로 벌한다."""
    lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=out.device).view(1, 1, 3, 3)
    def energy(t):
        g = (t[:, :1].float() + 1) * 0.5
        return F.avg_pool2d(F.conv2d(F.pad(g, (1, 1, 1, 1), mode='reflect'), lap) ** 2, k, 1, k // 2, count_include_pad=False)
    w = m_eff.detach().float()
    lmap = (torch.log(energy(out) + eps) - torch.log(energy(x) + eps)).abs()
    return ((lmap * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) + 1e-6)).mean()


def temporal_loss(o1, x1, o2, x2, has_nb, w_anchor=1.0):
    """인접 프레임 쌍(t, t') 출력 차이 제약. 두 프레임은 거의 같은 장면이므로 주입된 residual
    r = lowpass(I_out - I_in) (gray) 이 같아야 한다: Σ|r_t - r_t'| / Σ max(m_eff_t, m_eff_t')  (변화 영역 대비 정규화)
    + anchor 중심 L1 (mask 위치가 프레임마다 튀는 것 억제). has_nb=0 인 샘플 제외."""
    w = has_nb.float().view(-1)
    if w.sum() == 0:
        return x1.new_zeros(())
    lp = lambda r: F.avg_pool2d(F.avg_pool2d(r, 9, 1, 4, count_include_pad=False), 9, 1, 4, count_include_pad=False)
    r1 = lp((o1['out'].float() - x1.float())[:, :1])      # 저주파만 비교: 프레임 간 speckle 차이(줄일 수 없음)는 제외,
    r2 = lp((o2['out'].float() - x2.float())[:, :1])      # 보이는 flicker(mask 위치/밝기 변동)만 벌점
    u = torch.maximum(o1['m_eff'].float(), o2['m_eff'].float())
    l = (r1 - r2).abs().sum((1, 2, 3)) / (u.sum((1, 2, 3)) + 1.0)
    if w_anchor > 0 and o1.get('anchor') is not None:
        l = l + w_anchor * (o1['anchor'].float() - o2['anchor'].float()).abs().sum(1)
    return (l * w).sum() / w.sum()


# ------------------------------------------------------------------ combined
class GeneratorLoss(nn.Module):
    def __init__(self, i_vis, i_ani, gan_mode='lsgan', beta3=0.5, area_target=0.10, mask_smooth='l1', smooth_w=0.1,
                 cls_mode='ce', hinge_pt=0.7, hinge_ps=0.15, rec_region='full', i_abs=None, hinge_pb=0.0):
        super().__init__()
        self.cls_mode, self.hinge_pt, self.hinge_ps, self.rec_region = cls_mode, hinge_pt, hinge_ps, rec_region
        self.i_abs, self.hinge_pb = i_abs, hinge_pb
        self.mask_smooth, self.smooth_w = mask_smooth, smooth_w
        self.i_vis, self.i_ani, self.beta3, self.area_target = i_vis, i_ani, beta3, area_target
        self.gan = GANLoss(gan_mode)

    def cls_loss(self, logits):
        if self.cls_mode == 'hinge':
            return hinge_cls_loss(logits, self.i_vis, self.i_ani, self.hinge_pt, self.hinge_ps, self.i_abs, self.hinge_pb)
        return targeted_cls_loss(logits, self.i_vis, self.i_ani, self.beta3)

    def forward(self, g_out, d_fake, logits, rec_out, ref, has_ref, m_prior, lam, x_in=None, d_w=None):
        """lam: dict(cls, gan, rec, tv, area, mag). logits=None 이면 L_cls 생략(warmup).
        logits 가 list 이면 EOT 샘플별 L_cls 평균. d_w: masked PatchGAN 가중치."""
        L = {}
        if logits is None:
            L['cls'] = d_fake.new_zeros(())
        elif isinstance(logits, (list, tuple)):
            L['cls'] = torch.stack([self.cls_loss(z) for z in logits]).mean()
        else:
            L['cls'] = self.cls_loss(logits)
        L['gan'] = self.gan.g_loss(d_fake, d_w)
        L['rec'] = masked_rec_loss(rec_out, ref, has_ref, g_out['m_eff']) if self.rec_region == 'mask' \
            else rec_loss(rec_out, ref, has_ref)
        L['tv'] = masked_tv(g_out['delta'], g_out['m_eff'])
        sm = mask_grad_l2(g_out['m_hat']) if self.mask_smooth == 'l2' else mask_tv(g_out['m_hat'])
        L['area'] = mask_area_loss(g_out['m_hat'], m_prior, self.area_target) + self.smooth_w * sm
        L['compact'] = mask_compact_loss(g_out['m_eff'])
        if lam.get('seam', 0) > 0:
            L['seam'] = seam_loss(g_out['delta'], g_out['m_eff'])
        if x_in is not None:
            L['mag'] = mag_loss(g_out['out'], x_in)
        total = sum(lam.get(k, 0.0) * L[k] for k in L)
        return total, {k: v.detach().item() for k, v in L.items()}
