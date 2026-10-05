"""Nerve Visible -> Anisotropic Localized Residual GAN 학습 (Frozen 3-class classifier feedback).

Curriculum:
  ep 1..warmup            : λ_cls = 0            (L_rec + L_gan 중심, 자연스러운 anisotropic 질감)
  ep warmup+1..+ramp      : λ_cls 0 -> cls_max 선형 증가
  이후 (--adaptive)       : val 목표 미달 시 λ_cls *= 1.5 (최대 cls_cap)
목표: I_out 에 대해 3-class softmax P(ani) >= 0.85 and P(vis) <= 0.10

Usage:
  python classifier.py --split val          # (권장) 분류기 repeat/rederive 설정 사전 점검
  python train.py --epochs 40 --batch 16
"""
import argparse
import itertools
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from classifier import FrozenClassifier
from common import original_hw, rederive
from data import SourceDS, TargetDS
from losses import GANLoss, GeneratorLoss, feat_completion_loss, patch_weight, speckle_loss, temporal_loss
from models import PatchDiscriminator
from runcfg import build_generator, resolve, src_tgt_probs


def lam_cls_at(ep, a, state):
    if ep <= a.warmup:
        return 0.0
    if ep <= a.warmup + a.ramp:
        return a.cls_max * (ep - a.warmup) / a.ramp
    return state['cls']


def to_cls(img, a, ohw):
    return rederive(img, ohw) if a.cls_input == 'rederive' else img


def to_d(img, a):
    return img[:, :1] if a.d_ch == 1 else img


def ref_kw(b, a, dev, pre='', v=None):
    """과거 참조 프레임 kwargs (ref_in 일 때). pre='nb_' 이면 temporal 이웃 프레임 자신의 과거 참조. v: 유효 마스크 override."""
    if getattr(a, 'mem_k', 0):                             # 과거 feature 메모리 (v: [B] 샘플 단위 dropout 마스크)
        mv = b[pre + 'mem_v'].to(dev)
        if v is not None:
            mv = mv * v.view(-1, 1)
        return dict(mem=b[pre + 'mem'].to(dev, non_blocking=True), mem_m=b[pre + 'mem_mask'].to(dev, non_blocking=True),
                    mem_v=mv, mem_gap=b[pre + 'mem_gap'].to(dev))
    if not getattr(a, 'ref_in', 0):
        return {}
    return dict(ref=b[pre + 'pref'].to(dev, non_blocking=True), ref_m=b[pre + 'pref_mask'].to(dev, non_blocking=True),
                ref_v=b[pre + 'has_pref'].to(dev) if v is None else v, ref_gap=b[pre + 'pref_gap'].to(dev),
                **({'ref_flow': b[pre + 'pref_flow'].to(dev, non_blocking=True)} if pre + 'pref_flow' in b else {}))


def set_track(G, G2, b, a, dev, pre='', keep=None):
    """anchor 추적: 과거 가장 최근 target(VIS) 프레임(pref, ≤track_gap)을 역방향 G2(vis2ani) 에 넣어 나온 anchor 를
    G 의 heatmap Gaussian prior(σ=track_sigma) 로 설정. G2 결과 이미지는 쓰지 않음(no_grad, anchor 만). keep: [B] dropout 마스크."""
    if G2 is None or (pre + 'pref') not in b:
        G.anc_ref = None
        return None
    v = b[pre + 'has_pref'].to(dev).float()
    if keep is not None:
        v = v * keep
    tr = G2.training
    G2.eval()
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev.type == 'cuda'):
        G2.anc_ref = None
        r = G2(b[pre + 'pref'].to(dev, non_blocking=True), b[pre + 'pref_mask'].to(dev, non_blocking=True))['anchor'].float()
    G2.train(tr)
    G.anc_ref, G.anc_ref_valid, G.anc_ref_sigma = r, v, a.track_sigma
    return r


def reverse_ns(a):
    """추적용 역방향(vis2ani) 설정: 방향별 인자는 runcfg 기본값."""
    ns = argparse.Namespace(**vars(a))
    ns.direction = 'vis2ani' if a.direction == 'ani2vis' else 'ani2vis'
    for k in ('target_p', 'src_p', 'gain_range', 'max_offset', 'alpha_range'):
        setattr(ns, k, None)
    for k, v in resolve(ns).items():
        if k in ('target_p', 'src_p', 'gain_range', 'max_offset', 'alpha_range'):
            setattr(ns, k, v)
    return ns


def cls_logits(C, img, a, ohw):
    """EOT: eot 개 샘플(랜덤 flip/shift + 선택적 가우시안 노이즈) 의 logits list. eot<=1 이면 기존과 동일(1개)."""
    xc = to_cls(img, a, ohw)
    if a.eot <= 1:
        return C(xc, eot=bool(a.eot))
    zs = []
    for _ in range(a.eot):
        xi = xc + a.eot_noise * torch.randn_like(xc) if a.eot_noise > 0 else xc
        zs.append(C(xi, eot=True))
    return zs


def mem_consistency_loss(G, out, m, o, keep, a, ohw):
    """편집된 I_out 을 (CLAHE/scharr 재유도 후) plain encoder 로 다시 인코딩한 feature 가, 같은 위치에서
    공간 attention 으로 읽어 온 과거 VIS 메모리 R 과 방향이 같아지도록 (1-cos, target prior M_spec 가중, R detach).
    -> 메모리를 실제로 반영해야만 줄어드는 항 (기존 MemFuse 는 메모리 없이도 모든 loss 를 풀 수 있어 무시됨)."""
    xin = to_cls(out, a, ohw)
    ps = [p for p in G.enc.parameters() if p.requires_grad]
    for p in ps:                                           # gradient 는 I_out 으로만 (encoder 가 feature 를 붕괴시켜 loss 를 푸는 것 차단)
        p.requires_grad_(False)
    try:
        with torch.autocast('cuda', torch.bfloat16):
            col = G.plain_feats(xin, m)
    finally:
        for p in ps:
            p.requires_grad_(True)
    w = o['mem_valid'] * keep
    if w.sum() == 0:
        return out.new_zeros(())
    ls = []
    for i, r in o['mem_reads'].items():
        f = col[i].float()
        mm = F.interpolate(o['m_spec'].detach().float(), size=f.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        cos = F.cosine_similarity(f, r.detach().float(), dim=1)
        ls.append(((1 - cos) * mm).sum((1, 2)) / (mm.sum((1, 2)) + 1e-6))
    l = torch.stack(ls, 0).mean(0)
    return (l * w).sum() / w.sum()


def mem_align_loss(G, x, m, o, keep):
    """attention 대응 학습 (self-supervised tracking): 신경 prior 밖(M·(1-M_spec)) 조직은 현재·과거 프레임에 모두 보이므로,
    attention 으로 읽은 과거 feature 가 현재 프레임 plain feature 와 같아야 한다 (1-cos, target detach, read 는 q/k 로 grad).
    같은 q/k 가 신경 영역에도 쓰이므로 가려진 신경의 과거 대응 위치를 찾게 된다."""
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        col = G.plain_feats(x, m)
    w = o['mem_valid'] * keep
    if w.sum() == 0:
        return x.new_zeros(())
    ls = []
    for i, r in o['mem_reads'].items():
        f = col[i].float()
        sz = f.shape[-2:]
        mm = (F.interpolate(m.float(), size=sz, mode='area') *
              (1 - F.interpolate(o['m_spec'].detach().float(), size=sz, mode='bilinear', align_corners=False)))[:, 0]
        cos = F.cosine_similarity(f, r.float(), dim=1)
        ls.append(((1 - cos) * mm).sum((1, 2)) / (mm.sum((1, 2)) + 1e-6))
    l = torch.stack(ls, 0).mean(0)
    return (l * w).sum() / w.sum()


def select_score(res, a):
    """best 선택: 성공률 + 비용(목표 초과 면적, 수정 세기, flicker). sel_* = 0 이면 기존 기준."""
    return (res['success'] + 0.1 * res['pt'] - a.sel_area * max(0.0, res['area'] - a.area_target)
            - a.sel_dmag * res['dmag'] - a.sel_flick * res.get('flicker', 0.0))


@torch.no_grad()
def evaluate(G, C, loader, a, ohw, dev, save_png=None, G2=None):
    d = resolve(a)
    G.eval()
    acc = {k: [] for k in ('pv_in', 'pa_in', 'pv', 'pa', 'pb', 'ps', 'pt', 'area', 'dmag', 'hp')}
    tl, tn = 0.0, 0.0
    first = None
    ancs, ancs1 = [], []                                    # anchor 위치 (최종 창 / pass1 SCA prior), 정규화 (x, y)
    for b in loader:
        x, m = b['x'].to(dev), b['mask'].to(dev)
        with torch.autocast('cuda', torch.bfloat16, enabled=dev.type == 'cuda'):
            trk = set_track(G, G2, b, a, dev)
            o = G(x, m, **ref_kw(b, a, dev))
            if getattr(a, 'tex', 0) and 'has_pref' in b:   # tex 통계 (참조 유효 샘플, anchor 창 m_eff 가중 평균)
                fl, bh, bl = G._tex_stat
                wv = o['m_eff'].float() * (b['has_pref'].to(dev).float().view(-1, 1, 1, 1) > 0.5)
                ws = wv.sum() + 1e-6
                for k_, v_ in (('t_bhf', bh), ('t_blf', bl), ('t_flow', fl.float().norm(dim=1, keepdim=True) * x.shape[-1])):
                    acc.setdefault(k_, []).append(((v_.float() * wv).sum() / ws).view(1))
            lg_in = C(to_cls(x, a, ohw))
            lg = C(to_cls(o['out'].float(), a, ohw))
        pv0, pa0, _ = C.sem_probs(lg_in)
        ps, pt, pv, pa, pb = src_tgt_probs(C, lg, d)
        acc['ps'].append(ps); acc['pt'].append(pt)
        hp = (b['mem_v'].sum(1) > 0).float() if 'mem_v' in b else (b['has_pref'].float() if 'has_pref' in b else None)
        acc['hp'].append(hp.to(dev) if hp is not None else torch.zeros_like(pt))
        if 'nb' in b:                                      # temporal flicker (인접 프레임 residual 차이, 낮을수록 안정)
            hn = b['has_nb'].to(dev)
            if hn.sum() > 0:
                set_track(G, G2, b, a, dev, 'nb_')
                with torch.autocast('cuda', torch.bfloat16, enabled=dev.type == 'cuda'):
                    o2 = G(b['nb'].to(dev), b['nb_mask'].to(dev), **ref_kw(b, a, dev, 'nb_'))
                tl += temporal_loss(o, x, o2, b['nb'].to(dev), hn, 0.0).item() * hn.sum().item(); tn += hn.sum().item()
        me = o['m_eff'].float()
        acc['pv_in'].append(pv0); acc['pa_in'].append(pa0)
        acc['pv'].append(pv); acc['pa'].append(pa); acc['pb'].append(pb)
        acc['area'].append(me.sum((1, 2, 3)) / (m.sum((1, 2, 3)) + 1e-6))
        acc['dmag'].append((o['out'].float() - x).abs().sum((1, 2, 3)) / (me.sum((1, 2, 3)) * x.shape[1] + 1e-6))
        if trk is not None and o.get('anchor') is not None:   # 추적 prior 와 최종 anchor 거리 (prior 유효 샘플, px)
            vv = b['has_pref'].to(dev) > 0.5
            if vv.any():
                acc.setdefault('trk_d', []).append(((o['anchor'].float() - trk)[vv] * x.shape[-1]).norm(dim=1))
        if o.get('anchor') is not None:
            ancs.append(o['anchor'].float().cpu())
            if o.get('anchor1') is not None:
                ancs1.append(o['anchor1'].float().cpu())
        if first is None:
            heat = None
            if o.get('anc_logit') is not None:                # 최종 창 anchor heatmap (softmax, 온도 반영)
                h_, w_ = o['anc_hw']
                heat = torch.softmax(o['anc_logit'][:8].float() / getattr(G, 'anc_tau', 1.0), 1).view(-1, h_, w_).cpu()
            first = (x[:8].float().cpu(), o['out'][:8].float().cpu(), o['m_eff'][:8].float().cpu(), pt[:8].cpu(),
                     ancs[-1][:8] if ancs else None, ancs1[-1][:8] if ancs1 else None, heat)
    r = {k: torch.cat(v) for k, v in acc.items()}
    ok = ((r['pt'] >= d['target_p']) & (r['ps'] <= d['src_p'])).float()
    succ = ok.mean().item()
    res = {'success': succ, **{k: v.mean().item() for k, v in r.items() if k != 'hp'}}
    if 'trk_d' in r:
        res['trk_d'] = r['trk_d'].mean().item() if len(r['trk_d']) else float('nan')
    hp = r['hp'] > 0.5
    if hp.any():                                            # 과거 참조 유무별 성공률
        res['succ_ref'] = ok[hp].mean().item()
        res['succ_noref'] = ok[~hp].mean().item() if (~hp).any() else float('nan')
        res['ref_frac'] = hp.float().mean().item()
    if tn > 0:
        res['flicker'] = tl / tn
    if ancs:                                                # anchor 분포 (px, 생성기 해상도 기준)
        A = torch.cat(ancs)
        H, W = loader.dataset[0]['x'].shape[-2:] if first is None else first[0].shape[-2:]
        res['anc_mx'], res['anc_my'] = A[:, 0].mean().item() * W, A[:, 1].mean().item() * H
        res['anc_sx'], res['anc_sy'] = A[:, 0].std().item() * W, A[:, 1].std().item() * H
        if ancs1:
            A1 = torch.cat(ancs1)
            res['anc_d1'] = ((A - A1) * torch.tensor([W, H])).norm(dim=1).mean().item()   # |최종 - pass1| px
    if save_png and first is not None:
        x, out, me, pt, an, an1, heat = first
        H, W = x.shape[-2:]
        rows = []
        for i in range(x.shape[0]):
            g_in = ((x[i, 0] + 1) * 127.5).numpy()
            g_out = ((out[i, 0] + 1) * 127.5).numpy()
            diff = np.clip(128 + (g_out - g_in) * 2, 0, 255)
            mm = me[i, 0].numpy() * 255
            row = cv2.cvtColor(np.concatenate([g_in, g_out, diff, mm], 1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if heat is not None:                            # 5열: anchor heatmap (max 정규화 jet)
                hm = heat[i].numpy()
                hj = cv2.applyColorMap((hm / (hm.max() + 1e-12) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                hj = cv2.resize(hj, (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.putText(hj, f'pmax={hm.max():.3f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                row = np.concatenate([row, hj], 1)
            if an is not None:                              # 빨강 = 최종 anchor (원 반경 = anchor_r), 초록 + = pass1 anchor
                r_ = int(round(a.anchor_r * min(H, W)))
                for col in range(row.shape[1] // W):
                    cx, cy = int(an[i, 0] * W) + col * W, int(an[i, 1] * H)
                    cv2.circle(row, (cx, cy), 3, (0, 0, 255), -1)
                    if col < 2:
                        cv2.circle(row, (cx, cy), r_, (0, 0, 255), 1)
                    if an1 is not None:
                        cv2.drawMarker(row, (int(an1[i, 0] * W) + col * W, int(an1[i, 1] * H)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(row, f'P_{d["tgt_sem"][:3]}={pt[i]:.2f}', (W + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            rows.append(row)
        cv2.imwrite(save_png, np.concatenate(rows, 0))     # [input | output | diff(x2) | M_eff | anchor heat]
        if ancs:                                            # anc_epXXX.png: val 전체 anchor 산점도 (첫 입력 위)
            bg = cv2.cvtColor(((x[0, 0] + 1) * 127.5).numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR) // 2
            cv2.drawMarker(bg, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            if ancs1:
                for p_ in A1.numpy():
                    cv2.circle(bg, (int(p_[0] * W), int(p_[1] * H)), 1, (0, 255, 0), -1)
            for p_ in A.numpy():
                cv2.circle(bg, (int(p_[0] * W), int(p_[1] * H)), 1, (0, 0, 255), -1)
            txt = f'n={len(A)} mean=({res["anc_mx"]:.0f},{res["anc_my"]:.0f}) std=({res["anc_sx"]:.1f},{res["anc_sy"]:.1f})px'
            cv2.putText(bg, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            if 'anc_d1' in res:
                cv2.putText(bg, f'|final-pass1|={res["anc_d1"]:.1f}px  red=final green=pass1', (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.imwrite(save_png.replace('val_ep', 'anc_ep'), bg)
    G.anc_ref = None
    G.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--ramp', type=int, default=5)
    ap.add_argument('--cls_max', type=float, default=1.0)
    ap.add_argument('--cls_cap', type=float, default=8.0)
    ap.add_argument('--adaptive', type=int, default=1)
    ap.add_argument('--lam_gan', type=float, default=1.0)
    ap.add_argument('--lam_rec', type=float, default=5.0)
    ap.add_argument('--lam_tv', type=float, default=2.0)
    ap.add_argument('--lam_area', type=float, default=0.5)
    ap.add_argument('--lam_compact', type=float, default=0.2, help='마스크 compactness(단일 blob) 가중치')
    ap.add_argument('--area_target', type=float, default=0.10, help='prior 대비 허용 위치마스크 면적 비율')
    ap.add_argument('--beta3', type=float, default=0.5, help='3-class CE 보조항 가중치(0 = 명세 L_cls 만)')
    ap.add_argument('--gan_mode', choices=['lsgan', 'bce'], default='lsgan')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr_g', type=float, default=2e-4)
    ap.add_argument('--lr_d', type=float, default=2e-4)
    ap.add_argument('--max_delta', type=float, default=1.0)
    ap.add_argument('--anchor_r', type=float, default=0.0, help='>0: 단일 anchor window 반경(정규화 좌표), 0=off')
    ap.add_argument('--tex', type=int, default=0, help='1: 과거 실제 VIS 참조(--ref_in 필요)를 warp 해 HF 질감 + LF 잔차를 물리 맵 위에 더함')
    ap.add_argument('--enhance_only', type=int, default=0, help='1: 편집 ΔI >= 0 (밝히기만, 어둡게/지우기 금지)')
    ap.add_argument('--tex_fixed', type=int, default=0, help='1: tex warp flow 를 학습하지 않고 DIS optical flow(참조->현재 고정 정합) 사용')
    ap.add_argument('--tex_flow', type=float, default=0.15, help='tex warp flow 최대 크기 (이미지 폭 비율)')
    ap.add_argument('--detail_max', type=float, default=0.25, help='param=hybrid: 전해상도 detail residual 최대 크기')
    ap.add_argument('--param', choices=['physical', 'free', 'hybrid'], default='physical',
                    help='physical: /8 저주파 gain/offset/blur 맵, free: 픽셀별 ΔI')
    ap.add_argument('--repeat', type=int, default=4, help='정적 클립 GRU 반복 수 (classifier.py 로 점검)')
    ap.add_argument('--eot', type=int, default=1, help='L_cls 계산 시 랜덤 flip/shift 샘플 수 (0=off, 1=기존, N>1=N개 평균)')
    ap.add_argument('--eot_noise', type=float, default=0.0, help='EOT 샘플별 가우시안 노이즈 σ (eot>1 일 때)')
    ap.add_argument('--cls_mode', choices=['ce', 'hinge'], default='ce',
                    help='ce: 기존 targeted CE (계속 밀어냄), hinge: P(tgt)>=hinge_pt & P(src)<=hinge_ps 도달 시 gradient 0')
    ap.add_argument('--hinge_pt', type=float, default=0.65)
    ap.add_argument('--hinge_ps', type=float, default=0.15)
    ap.add_argument('--hinge_pb', type=float, default=0.0, help='>0 (hinge 모드): P(abs)<=hinge_pb 벌점 추가')
    ap.add_argument('--d_masked', type=int, default=0, help='1: fake 쪽 PatchGAN 을 수정된 patch(m_eff) 로만 판정')
    ap.add_argument('--rec_region', choices=['full', 'mask'], default='full', help='mask: 수정 영역에서만 pseudo-pair 비교')
    ap.add_argument('--lam_seam', type=float, default=0.0, help='마스크 경계 띠에서 residual gradient 벌점(뚜렷한 편집 경계 억제)')
    ap.add_argument('--detail_gate', type=int, default=0, help='1: hybrid detail 에 m_hat 을 한 번 더 곱해 경계 전에 소멸')
    ap.add_argument('--lam_mag', type=float, default=0.0, help='최소 수정 L1 (|I_out - I_in| 전체 평균) 가중치')
    ap.add_argument('--sel_area', type=float, default=0.0, help='best 선택 시 (area - area_target)+ 벌점')
    ap.add_argument('--sel_dmag', type=float, default=0.0, help='best 선택 시 |Δ| 벌점')
    ap.add_argument('--sel_flick', type=float, default=0.0, help='best 선택 시 flicker 벌점')
    ap.add_argument('--cls_input', choices=['rederive', 'direct'], default='rederive')
    ap.add_argument('--d_ch', type=int, choices=[1, 3], default=1)
    ap.add_argument('--max_gap', type=int, default=5, help='pseudo-pair 최대 프레임 간격')
    ap.add_argument('--mask_dir', default=None, help='외부 nerve mask 폴더(없으면 조직 영역 prior)')
    ap.add_argument('--direction', choices=['vis2ani', 'ani2vis'], default='vis2ani')
    ap.add_argument('--target_p', type=float, default=None, help='성공: P(target)>=target_p (기본: 방향별, runcfg.DEFAULTS)')
    ap.add_argument('--src_p', type=float, default=None, help='성공: P(source)<=src_p (기본: 방향별)')
    ap.add_argument('--gain_range', type=float, nargs=2, default=None, help='physical gain 범위 (기본: 방향별)')
    ap.add_argument('--max_offset', type=float, default=None)
    ap.add_argument('--alpha_range', type=float, nargs=2, default=None, help='blur-mix α 범위, 음수=unsharp')
    ap.add_argument('--sca_at', type=int, nargs='*', default=[],
                    help='Spatial Completion Adapter 삽입 encoder stride (예: 4 8 16 = shallow/middle/deep). 빈 값=off')
    ap.add_argument('--sca_iters', type=int, default=3, help='SCA 내부 gate/feat 반복 수')
    ap.add_argument('--sca_prior', choices=['anchor', 'mask'], default='anchor',
                    help='pass1 target prior: anchor=M⊙anchor window(box prompt), mask=M⊙m̂(pass1)')
    ap.add_argument('--cls_ckpt2', default=None, help='두 번째 frozen 분류기 (L_cls 를 두 분류기 평균으로: 단일 분류기 지름길 억제)')
    ap.add_argument('--lam_speckle', type=float, default=0.0, help='편집 영역 국소 Laplacian 에너지 로그비 (speckle 보존)')
    ap.add_argument('--lam_feat', type=float, default=0.0,
                    help='SCA feature completion loss (pseudo-pair target 프레임 plain feature 와 1-cos, M_spec 내부)')
    ap.add_argument('--up_after', type=int, default=0, help='1: physical 맵을 /8 에서 sigmoid/tanh 후 업샘플 (계단 경계 방지)')
    ap.add_argument('--blur_sigma', type=float, default=3.0, help='physical blur-mix 가우시안 σ(px). 해상도에 비례해 스케일 (224:3)')
    ap.add_argument('--phys_stride', type=int, choices=[8, 4, 2, 1], default=8, help='physical 맵 해상도: 8=up4·up3(/8), 4=+up2(/4), 2=+up1(/2), 1=+up0(/1, 전 skip U-Net)')
    ap.add_argument('--mask_head', type=int, default=0, help='1: Gaussian anchor 창 제거, [F28, anchor heatmap] -> 28² soft mask head (anchor_r>0 필요)')
    ap.add_argument('--encoder', choices=['small', 'large'], default='small', help='MobileNetV3 encoder 크기')
    ap.add_argument('--phys_blur', type=float, default=0.0, help='>0: 물리 맵 가우시안 평활 σ(셀 단위, up_after 필요). 계단/벌집 방지')
    ap.add_argument('--mask_smooth', choices=['l1', 'l2'], default='l1', help='mask 평활: l1=TV(기존), l2=gradient²')
    ap.add_argument('--smooth_w', type=float, default=0.1, help='mask 평활 항 가중치 (area 항 내부, λ_area 곱해짐)')
    ap.add_argument('--lam_temp', type=float, default=0.0, help='인접 프레임 출력 차이(temporal consistency) 가중치')
    ap.add_argument('--tc_gap', type=int, default=2, help='temporal 쌍 최대 프레임 간격 (같은 비디오, 같은 source 클래스)')
    ap.add_argument('--tc_start', type=int, default=-1,
                    help='temporal loss 시작 epoch (-1 = warmup 이후, λcls 와 함께). warmup 중엔 identity 가 자명한 해라 mask 붕괴')
    ap.add_argument('--tc_anchor', type=float, default=1.0, help='temporal loss 내 anchor 중심 L1 가중치')
    ap.add_argument('--ref_in', type=int, default=0, help='1: 과거 참조 프레임(가장 최근 target 클래스 프레임) 을 G 에 입력')
    ap.add_argument('--ref_gap', type=int, default=30, help='과거 참조 최대 프레임 거리')
    ap.add_argument('--ref_drop', type=float, default=0.2, help='학습 시 참조를 끄는 확률 (참조 없는 프레임 대비)')
    ap.add_argument('--mem_k', type=int, default=0, help='>0: 과거 K 프레임 feature 메모리 (분류기 P(vis) 게이트, GT 미사용)')
    ap.add_argument('--mem_gap', type=int, default=30, help='메모리 최대 과거 프레임 거리')
    ap.add_argument('--mem_thr', type=float, default=0.5, help='메모리 편입 기준 P(vis) (원본 입력)')
    ap.add_argument('--mem_mode', choices=['slot', 'xattn', 'sam'], default='slot',
                    help='slot: 기존 MemFuse(같은 위치 K-slot), xattn: 공간 cross-attention(위치 이동 대응)')
    ap.add_argument('--mem_at', type=int, nargs='*', default=[8, 16], help='xattn 메모리 삽입 encoder stride')
    ap.add_argument('--lam_memc', type=float, default=0.0,
                    help='메모리 일관성: 재인코딩한 I_out feature 가 메모리 read-out 과 같아지도록 (1-cos, M_spec 내부)')
    ap.add_argument('--mem_dist', type=int, default=1, help='xattn 거리 prior (0 = 위치 무관 content attention)')
    ap.add_argument('--lam_align', type=float, default=0.0,
                    help='메모리 대응 학습: 신경 prior(M_spec) *밖* 조직에서 attention read-out 이 현재 프레임 plain feature 를 복원 '
                         '(q/k 로 gradient -> 실제 대응점 추적 학습; 안쪽은 memc 로 VIS feature 주입)')
    ap.add_argument('--mem_drop', type=float, default=0.2, help='학습 시 메모리 전체를 끄는 확률')
    ap.add_argument('--init_from', default=None, help='G 가중치 초기화 체크포인트 (SCA 는 identity 로 추가)')
    ap.add_argument('--lam_ptv', type=float, default=0.0, help='/8 물리 맵(gain/offset/alpha, 평활 전) TV(L1) 벌점 -> 셀 단위 블록 억제')
    ap.add_argument('--lam_psat', type=float, default=0.0, help='/8 물리 맵 포화 벌점: [-1,1] 정규화 값 |z|>0.8 hinge² (범위 끝 bang-bang 억제)')
    ap.add_argument('--tgt_in', type=int, default=0,
                    help='1: target 클래스 프레임(ani2vis 의 VIS)도 G 입력. 더 확실한 target 으로(hinge P(tgt)>=tgt_pt) + 최소 수정, '
                         'temporal 쌍이 클래스 경계를 넘음 -> 경계에서 편집이 갑자기 켜지/꺼지는 현상 완화')
    ap.add_argument('--anc_dir', default=None, help='anchor 위치 의사 GT 마스크 폴더 (SAM2 신경 track, 예: track_masks_rp10)')
    ap.add_argument('--lam_anc', type=float, default=0.0, help='anchor soft-argmax L1 (중앙 편향 해소)')
    ap.add_argument('--anc_ce', type=float, default=0.1, help='anchor heatmap CE (Gaussian 목표, σ=anchor_r/2) 가중 (lam_anc 에 곱함)')
    ap.add_argument('--lam_asearch', type=float, default=0.0,
                    help='anchor search: pass1(SCA prior) anchor 후보 K 개를 실제 G+분류기로 평가해 가장 성능 좋은 위치로 heatmap 학습')
    ap.add_argument('--lam_acons', type=float, default=0.0,
                    help='anchor 일관성: 최종 편집 창 anchor 를 pass1(SCA prior) anchor(detach)에 맞춤 (L1 + 0.1 heatmap KL)')
    ap.add_argument('--anc_local', type=int, default=0, help='>0: anchor = argmax 셀 주변 (2k+1)² 국소 soft-argmax (중앙 수축 방지)')
    ap.add_argument('--anc_pass1', type=float, default=0.0, help='anchor 지도(lam_anc)를 pass1(SCA prior) anchor 에도 이 가중으로')
    ap.add_argument('--anc_tau', type=float, default=1.0, help='anchor heatmap softmax 온도 (<1 이면 날카로운 soft-argmax)')
    ap.add_argument('--as_sigma', type=float, default=0.0, help='anchor search/지도 heatmap 목표 Gaussian σ (0: anchor_r/2, 예 0.02=약 1셀)')
    ap.add_argument('--as_k', type=int, default=4, help='anchor search 후보 수 (현재 anchor 포함)')
    ap.add_argument('--as_n', type=int, default=8, help='step 당 anchor search 적용 source 샘플 수 상한')
    ap.add_argument('--tgt_frac', type=float, default=0.3, help='tgt_in: 배치 중 target 클래스 item 기대 비율')
    ap.add_argument('--tgt_pt', type=float, default=0.8, help='tgt_in: target 입력의 hinge 목표 P(tgt)')
    ap.add_argument('--lam_tgt_cls', type=float, default=1.0, help='tgt_in: hinge 가중 (λcls 스케줄에 곱함)')
    ap.add_argument('--lam_tgt_mag', type=float, default=10.0, help='tgt_in: target 입력의 |ΔI| 이미지 평균 벌점')
    ap.add_argument('--track', type=int, default=0,
                    help='1: 역방향 G2(vis2ani)를 함께 학습, 과거 최근 VIS 프레임(≤track_gap)의 G2 anchor 를 G anchor heatmap prior 로 (추적용, G2 출력 이미지는 미사용)')
    ap.add_argument('--track_gap', type=int, default=30, help='추적 참조 VIS 프레임 최대 과거 거리')
    ap.add_argument('--track_sigma', type=float, default=0.08, help='추적 prior Gaussian σ (정규화 좌표)')
    ap.add_argument('--track_drop', type=float, default=0.2, help='학습 시 추적 prior 를 끄는 확률 (참조 없는 프레임 대비)')
    ap.add_argument('--train_split', default='train', help="생성기 학습 split ('train+val' 로 합치기 가능)")
    ap.add_argument('--val_split', default='val', help='모델 선택/검증 split (val 을 train 에 편입했으면 test)')
    ap.add_argument('--val_n', type=int, default=600)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--out', default='runs/aniso_gan')
    ap.add_argument('--resume', action='store_true', help='<out>/last.pt 에서 이어서 학습 (G/D/opt/scheduler/state/best 복원)')
    a = ap.parse_args()
    from common import CLS_CKPT, SIZE
    a.size, a.cls_ckpt = SIZE, CLS_CKPT                     # 해상도·분류기 기록 (ANSIO_SIZE / ANSIO_CLS_CKPT)
    for k, v in resolve(a).items():                         # 방향별 기본값 채워서 args.json/ckpt 에 기록
        if k in ('target_p', 'src_p', 'gain_range', 'max_offset', 'alpha_range'):
            setattr(a, k, v)
    d = resolve(a)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=2)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    ohw = original_hw()

    # ---------------- data
    src, tgt = d['src_sem'], d['tgt_sem']
    nbg = a.tc_gap if a.lam_temp > 0 else 0
    pg = a.ref_gap if a.ref_in else (a.track_gap if a.track else 0)
    mk = dict(mem_k=a.mem_k, mem_gap=a.mem_gap, mem_thr=a.mem_thr)
    ds_tr = SourceDS(a.train_split, True, a.max_gap, a.mask_dir, src=src, ref=tgt, nb_gap=nbg, pref_gap=pg, tgt_in=bool(a.tgt_in), anc_dir=a.anc_dir if a.lam_anc > 0 else None, ref_flow=bool(a.tex_fixed), **mk)
    ds_va = SourceDS(a.val_split, False, a.max_gap, a.mask_dir, src=src, ref=tgt, nb_gap=a.tc_gap, pref_gap=pg, ref_flow=bool(a.tex_fixed), **mk)
    if a.mem_k:
        nm = [len(ds_tr.mem_of.get(it[0], [])) for it in ds_tr.items]
        print(f'memory K={a.mem_k} gap≤{a.mem_gap} P(vis)≥{a.mem_thr}: train src with mem={np.mean([n > 0 for n in nm]):.3f} '
              f'mean slots={np.mean(nm):.2f}', flush=True)
    n_pref_va = ds_va.n_pref
    if a.val_n and len(ds_va) > a.val_n:
        ds_va = Subset(ds_va, np.linspace(0, len(ds_va) - 1, a.val_n).astype(int).tolist())
    sampler = None
    if a.tgt_in:                                            # source item 수는 유지, target item 이 기대 비율 tgt_frac 이 되도록 샘플링
        isr = np.array(ds_tr.is_src, bool); ns, nt = isr.sum(), (~isr).sum()
        wts = np.where(isr, (1 - a.tgt_frac) / ns, a.tgt_frac / nt)
        sampler = torch.utils.data.WeightedRandomSampler(wts, int(ns / (1 - a.tgt_frac)), replacement=True)
        print(f'tgt_in: train src={ns} tgt={nt} frac={a.tgt_frac} -> {len(sampler)} samples/epoch', flush=True)
        ds_vt = SourceDS(a.val_split, False, a.max_gap, a.mask_dir, src=src, ref=tgt, nb_gap=a.tc_gap, tgt_in=True,
                         pref_gap=pg, ref_flow=bool(a.tex_fixed))   # ref_in: target 입력도 과거 참조 필요
        ti = [k for k, v in enumerate(ds_vt.is_src) if not v]
        dl_vt = DataLoader(Subset(ds_vt, [ti[j] for j in np.linspace(0, len(ti) - 1, min(a.val_n or len(ti), len(ti))).astype(int)]),
                           32, shuffle=False, num_workers=a.workers, pin_memory=True)
    dl_tr = DataLoader(ds_tr, a.batch, shuffle=sampler is None, sampler=sampler, num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, 32, shuffle=False, num_workers=a.workers, pin_memory=True)
    dl_an = DataLoader(TargetDS(a.train_split, True, cls=tgt), a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                       drop_last=True, persistent_workers=a.workers > 0)
    it_an = itertools.cycle(dl_an) if len(dl_an) else None
    print(f'direction={a.direction} ({src} -> {tgt}) | train src={len(ds_tr)} (paired={ds_tr.n_paired}) val={len(ds_va)} '
          f'temporal pairs train={ds_tr.n_nb} | past-ref(≤{a.ref_gap}) train={ds_tr.n_pref} val={n_pref_va} | real tgt={len(dl_an.dataset)} | success: P({tgt})>={d["target_p"]} & P({src})<={d["src_p"]} | '
          f'gain={d["gain_range"]} offset=±{d["max_offset"]} alpha={d["alpha_range"]} orig_hw={ohw}', flush=True)

    # ---------------- models
    C = FrozenClassifier(repeat=a.repeat).to(dev)
    C2 = FrozenClassifier(a.cls_ckpt2, repeat=a.repeat).to(dev) if a.cls_ckpt2 else None
    if C2 is not None:
        assert C2.classes == C.classes, (C2.classes, C.classes)
        print(f'classifier2 {a.cls_ckpt2} in_size={C2.in_size}', flush=True)
    print(f'classifier classes={C.classes} -> vis={C.i_vis} ani={C.i_ani} abs={C.i_abs}', flush=True)
    G = build_generator(d, pretrained=not a.no_pretrained).to(dev)
    if a.init_from:
        miss, unexp = G.load_state_dict(torch.load(a.init_from, map_location='cpu', weights_only=False)['G'], strict=False)
        print(f'init_from={a.init_from} missing={len(miss)} (SCA 등) unexpected={len(unexp)}', flush=True)
    D = PatchDiscriminator(in_ch=a.d_ch).to(dev)
    print(f'G params={sum(p.numel() for p in G.parameters())/1e6:.2f}M  '
          f'D params={sum(p.numel() for p in D.parameters())/1e6:.2f}M', flush=True)
    opt_g = torch.optim.Adam(G.parameters(), a.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), a.lr_d, betas=(0.5, 0.999))
    sch = lambda o: torch.optim.lr_scheduler.LambdaLR(
        o, lambda e: 1.0 if e < a.epochs // 2 else max(0.0, 1 - (e - a.epochs // 2) / (a.epochs - a.epochs // 2 + 1)))
    sch_g, sch_d = sch(opt_g), sch(opt_d)
    i_sem = {'visible': C.i_vis, 'anisotropic': C.i_ani}
    crit_g = GeneratorLoss(i_sem[src], i_sem[tgt], a.gan_mode, a.beta3, a.area_target, a.mask_smooth, a.smooth_w,
                           a.cls_mode, a.hinge_pt, a.hinge_ps, a.rec_region, C.i_abs, a.hinge_pb)   # L_cls: source -> target
    crit_d = GANLoss(a.gan_mode)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda')
    G2 = D2 = None
    if a.track:                                             # 역방향 G2 (추적용 anchor 공급), 자체 GAN/분류 손실로 함께 학습
        a2 = reverse_ns(a); d2 = resolve(a2)
        G2 = build_generator(d2, pretrained=not a.no_pretrained).to(dev)
        D2 = PatchDiscriminator(in_ch=a.d_ch).to(dev)
        opt_g2 = torch.optim.Adam(G2.parameters(), a.lr_g, betas=(0.5, 0.999))
        opt_d2 = torch.optim.Adam(D2.parameters(), a.lr_d, betas=(0.5, 0.999))
        sch_g2, sch_d2 = sch(opt_g2), sch(opt_d2)
        crit_g2 = GeneratorLoss(i_sem[tgt], i_sem[src], a.gan_mode, a.beta3, a.area_target, a.mask_smooth, a.smooth_w,
                                a.cls_mode, a.hinge_pt, a.hinge_ps, a.rec_region, C.i_abs, a.hinge_pb)
        ds2_tr = SourceDS(a.train_split, True, a.max_gap, a.mask_dir, src=tgt, ref=src, nb_gap=nbg)
        ds2_va = SourceDS(a.val_split, False, a.max_gap, a.mask_dir, src=tgt, ref=src, nb_gap=a.tc_gap)
        if a.val_n and len(ds2_va) > a.val_n:
            ds2_va = Subset(ds2_va, np.linspace(0, len(ds2_va) - 1, a.val_n).astype(int).tolist())
        it2 = itertools.cycle(DataLoader(ds2_tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                                         persistent_workers=a.workers > 0))
        dl2_va = DataLoader(ds2_va, 32, shuffle=False, num_workers=a.workers, pin_memory=True)
        it_an2 = itertools.cycle(DataLoader(TargetDS(a.train_split, True, cls=src), a.batch, shuffle=True, num_workers=a.workers,
                                            pin_memory=True, drop_last=True, persistent_workers=a.workers > 0))
        print(f'track: G2 {a2.direction} train src={len(ds2_tr)} val={len(ds2_va)} | gap≤{a.track_gap} σ={a.track_sigma} '
              f'drop={a.track_drop} | G past-VIS ref train={ds_tr.n_pref} val={n_pref_va}', flush=True)
    state2 = {'cls': a.cls_max}

    base = evaluate(G, C, dl_va, a, ohw, dev)
    print(f'[input baseline] P(vis)={base["pv_in"]:.3f} P(ani)={base["pa_in"]:.3f}', flush=True)

    state = {'cls': a.cls_max}
    best = -1.0
    start = 1
    if a.resume:
        ck = torch.load(os.path.join(a.out, 'last.pt'), map_location='cpu', weights_only=False)
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D'])
        opt_g.load_state_dict(ck['opt_g']); opt_d.load_state_dict(ck['opt_d'])
        state, start = ck['state'], ck['epoch'] + 1
        if G2 is not None and 'G2' in ck:
            G2.load_state_dict(ck['G2']); D2.load_state_dict(ck['D2'])
            opt_g2.load_state_dict(ck['opt_g2']); opt_d2.load_state_dict(ck['opt_d2']); state2 = ck['state2']
            for _ in range(ck['epoch']):
                sch_g2.step(); sch_d2.step()
        for _ in range(ck['epoch']):
            sch_g.step(); sch_d.step()
        bp = os.path.join(a.out, 'G_best.pt')
        if os.path.exists(bp):
            best = select_score(torch.load(bp, map_location='cpu', weights_only=False)['val'], a)
        print(f'resume from {a.out}/last.pt epoch={ck["epoch"]} state={state} best={best:.4f} lr_g={opt_g.param_groups[0]["lr"]:.2e}', flush=True)
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    for ep in range(start, a.epochs + 1):
        tc_on = a.lam_temp > 0 and ep > (a.warmup if a.tc_start < 0 else a.tc_start - 1)
        tc_w = a.lam_temp * min(1.0, (ep - (a.warmup if a.tc_start < 0 else a.tc_start - 1)) / max(a.ramp, 1)) if tc_on else 0.0   # λcls 와 같은 속도로 ramp
        lam = dict(cls=lam_cls_at(ep, a, state), gan=a.lam_gan, rec=a.lam_rec, tv=a.lam_tv, area=a.lam_area, compact=a.lam_compact, seam=a.lam_seam,
                   mag=a.lam_mag * min(1.0, max(0, ep - a.warmup) / max(a.ramp, 1)))   # warmup 중엔 identity 가 자명해 off, λcls 와 같이 ramp
        lam2 = dict(lam, cls=lam_cls_at(ep, a, state2))
        G.train(); D.train(); t0 = time.time()
        if G2 is not None:
            G2.train(); D2.train()
        agg, n = {}, 0
        for b in dl_tr:
            x, m = b['x'].to(dev, non_blocking=True), b['mask'].to(dev, non_blocking=True)
            ref, has = b['ref'].to(dev, non_blocking=True), b['has_ref'].to(dev, non_blocking=True)
            xa = next(it_an).to(dev, non_blocking=True) if it_an else ref

            # ---- G forward
            use_feat = a.lam_feat > 0 and bool(a.sca_at)
            with torch.autocast(**amp):
                if a.mem_k:                                    # 메모리 dropout: 이웃 프레임과 같은 마스크 (temporal 일관성)
                    keep = (torch.rand(x.shape[0], device=dev) >= a.mem_drop).float()
                    rk, rk_nb = ref_kw(b, a, dev, v=keep), (ref_kw(b, a, dev, 'nb_', v=keep) if tc_on else {})
                elif a.ref_in:                                 # 참조 dropout: 이웃 프레임과 같은 마스크 (temporal 일관성)
                    keep = (torch.rand(x.shape[0], device=dev) >= a.ref_drop).float()
                    rk, rk_nb = ref_kw(b, a, dev, v=b['has_pref'].to(dev) * keep), \
                        (ref_kw(b, a, dev, 'nb_', v=b['nb_has_pref'].to(dev) * keep) if tc_on else {})
                else:
                    rk, rk_nb = {}, {}
                tkeep = (torch.rand(x.shape[0], device=dev) >= a.track_drop).float()   # 추적 prior dropout (이웃 프레임과 같은 마스크)
                trk = set_track(G, G2, b, a, dev, keep=tkeep)
                o = G(x, m, keep_sca=use_feat, **rk)
                tgt_f = [t.detach() for t in G.sca_targets(ref, m)] if use_feat else None   # target detach: encoder 붕괴로 자명해 방지
                if tc_on:
                    set_track(G, G2, b, a, dev, 'nb_', keep=tkeep)
                o_nb = G(b['nb'].to(dev, non_blocking=True), b['nb_mask'].to(dev, non_blocking=True), **rk_nb) if tc_on else None
                G.anc_ref = None
            out = o['out'].float()

            # ---- D step
            with torch.autocast(**amp):
                df_d = D(to_d(out.detach(), a))
                ld = crit_d.d_loss(D(to_d(xa, a)), df_d, patch_weight(o['m_eff'], df_d) if a.d_masked else None)
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()

            # ---- G step
            with torch.autocast(**amp):
                d_fake = D(to_d(out, a))
                logits = cls_logits(C, out, a, ohw) if lam['cls'] > 0 else None
                if logits is not None and C2 is not None:           # 앙상블: 두 분류기 L_cls 평균
                    logits = [logits, cls_logits(C2, out, a, ohw)]
            if a.tgt_in:                                       # L_cls 는 source item 만, target item 은 hinge + 최소 수정
                isrc = b['is_src'].to(dev) > 0.5
                logits_all = logits
                sub = lambda z: [t[isrc] for t in z] if isinstance(z, (list, tuple)) else z[isrc]
                logits = sub(logits) if (logits is not None and isrc.any()) else None
            rec_o, rec_r = (out[:, :1], ref[:, :1]) if a.cls_input == 'rederive' else (out, ref)
            lg, parts = crit_g(o, d_fake, logits, rec_o, rec_r, has, m, lam, x_in=x if a.lam_mag > 0 else None,
                               d_w=patch_weight(o['m_eff'], d_fake) if a.d_masked else None)
            if a.tgt_in and (~isrc).any():
                it_ = ~isrc
                lmag = (out[it_] - x[it_].float()).abs().mean()
                lg = lg + a.lam_tgt_mag * lmag
                parts['tmag'] = lmag.item()
                if logits_all is not None:
                    zs = logits_all if isinstance(logits_all, (list, tuple)) else [logits_all]
                    lh = torch.stack([F.relu(np.log(a.tgt_pt) - F.log_softmax(z[it_].float(), -1)[:, i_sem[tgt]]).mean() for z in zs]).mean()
                    lg = lg + lam['cls'] * a.lam_tgt_cls * lh
                    parts['tcls'] = lh.item()
            if a.lam_anc > 0 and o.get('anc_logit') is not None and 'anc' in b:   # anchor 위치 지도학습 (신경 track 무게중심)
                ha, ca = b['has_anc'].to(dev), b['anc'].to(dev)
                if ha.sum() > 0:
                    l1 = ((o['anchor'].float() - ca).abs().sum(1) * ha).sum() / ha.sum()
                    h_, w_ = o['anc_hw']
                    yy = (torch.arange(h_, device=dev) + 0.5) / h_; xx = (torch.arange(w_, device=dev) + 0.5) / w_
                    d2 = (yy.view(1, -1, 1) - ca[:, 1].view(-1, 1, 1)) ** 2 + (xx.view(1, 1, -1) - ca[:, 0].view(-1, 1, 1)) ** 2
                    tq = torch.softmax(-d2.flatten(1) / (2 * (a.as_sigma or a.anchor_r / 2) ** 2), 1)
                    ce = (-(tq * F.log_softmax(o['anc_logit'].float(), 1)).sum(1) * ha).sum() / ha.sum()
                    lg = lg + a.lam_anc * (l1 + a.anc_ce * ce)
                    if a.anc_pass1 > 0 and o.get('anchor1') is not None:  # SCA prior 위치도 같은 목표
                        l1p = ((o['anchor1'].float() - ca).abs().sum(1) * ha).sum() / ha.sum()
                        cep = (-(tq * F.log_softmax(o['anc_logit1'].float(), 1)).sum(1) * ha).sum() / ha.sum()
                        lg = lg + a.lam_anc * a.anc_pass1 * (l1p + a.anc_ce * cep)
                    parts['anc'], parts['anc_ce'] = l1.item(), ce.item()
            if a.lam_acons > 0 and o.get('anchor1') is not None:      # SCA 보완 위치 = 실제 편집 위치 가 되도록
                l1c = (o['anchor'].float() - o['anchor1'].detach().float()).abs().sum(1).mean()
                kl = F.kl_div(F.log_softmax(o['anc_logit'].float(), 1), F.softmax(o['anc_logit1'].detach().float(), 1), reduction='batchmean')
                lg = lg + a.lam_acons * (l1c + 0.1 * kl)
                parts['acons'] = l1c.item()
            if a.lam_asearch > 0 and lam['cls'] > 0 and o.get('anc_logit1') is not None:
                # anchor search: SAM2 등 외부 위치 대신, pass1 anchor 후보들(현재 + heatmap/조직 균일 샘플)을 강제로 넣어
                # G(2-pass) + 분류기로 실제 결과(log P(tgt) - log P(src))를 비교 -> 가장 좋은 후보 위치를 pass1/최종 heatmap 의 목표로.
                sel = (b['is_src'].to(dev) > 0.5) if a.tgt_in else torch.ones(x.shape[0], dtype=torch.bool, device=dev)
                idx = sel.nonzero().flatten()[:a.as_n]
                if len(idx):
                    h_, w_ = o['anc_hw']; n_ = len(idx)
                    yy = (torch.arange(h_, device=dev) + 0.5) / h_; xx = (torch.arange(w_, device=dev) + 0.5) / w_
                    with torch.no_grad():
                        lg1 = o['anc_logit1'][idx].float()
                        tis = F.adaptive_avg_pool2d(m[idx].float(), (h_, w_)).flatten(1) + 1e-6
                        q = 0.5 * torch.softmax(lg1, 1) + 0.5 * tis / tis.sum(1, keepdim=True)
                        smp = torch.multinomial(q, a.as_k - 1, replacement=True)                       # [n, K-1]
                        cs = torch.stack([xx[smp % w_], yy[smp // w_]], -1)                            # [n, K-1, 2]
                        cands = torch.cat([o['anchor1'][idx].float()[:, None], cs], 1)                  # [n, K, 2] (0 = 현재)
                        Rw = []
                        for k in range(a.as_k):
                            with torch.autocast(**amp):
                                ok_ = G(x[idx], m[idx], anchor_c=cands[:, k])
                                lp = F.log_softmax(C(to_cls(ok_['out'].float(), a, ohw), eot=False).float(), -1)
                            Rw.append(lp[:, i_sem[tgt]] - lp[:, i_sem[src]])
                        Rw = torch.stack(Rw, 1)                                                         # [n, K]
                        best_k = Rw.argmax(1)
                        cb = cands[torch.arange(n_, device=dev), best_k]                                # [n, 2]
                        d2 = (yy.view(1, -1, 1) - cb[:, 1].view(-1, 1, 1)) ** 2 + (xx.view(1, 1, -1) - cb[:, 0].view(-1, 1, 1)) ** 2
                        sg = a.as_sigma or a.anchor_r / 2
                        tq = torch.softmax(-d2.flatten(1) / (2 * sg ** 2), 1)
                    ce = lambda z: -(tq * F.log_softmax(z[idx].float() / a.anc_tau, 1)).sum(1).mean()
                    las = (o['anchor1'][idx].float() - cb).abs().sum(1).mean() + 0.1 * (ce(o['anc_logit1']) + ce(o['anc_logit']))
                    lg = lg + a.lam_asearch * las
                    parts['as'] = las.item()
                    parts['as_gain'] = (Rw.max(1).values - Rw[:, 0]).mean().item()                  # 최선 후보 - 현재 anchor 의 보상 차
                    parts['as_move'] = (best_k > 0).float().mean().item()
            if (a.lam_ptv > 0 or a.lam_psat > 0) and o.get('phys8') is not None:
                q8 = o['phys8'][:, :3].float()             # s_gain, tanh(off), s_alpha
                w8 = F.adaptive_avg_pool2d(o['m_eff'].detach().float(), q8.shape[-2:]) + 0.05   # 편집 영역 위주
                tv8 = ((q8[..., 1:, :] - q8[..., :-1, :]).abs() * w8[..., 1:, :]).mean() + \
                      ((q8[..., :, 1:] - q8[..., :, :-1]).abs() * w8[..., :, 1:]).mean()
                z = torch.cat([2 * q8[:, 0:1] - 1, q8[:, 1:2], 2 * q8[:, 2:3] - 1], 1)
                sat = (F.relu(z.abs() - 0.8) ** 2 * w8).mean()
                lg = lg + a.lam_ptv * tv8 + a.lam_psat * sat
                parts['ptv'], parts['psat'] = tv8.item(), sat.item()
            if use_feat:
                lf = feat_completion_loss(o['sca_feats'], tgt_f, o['m_spec'], has)
                lg = lg + a.lam_feat * lf
                parts['feat'] = lf.item()
            if a.lam_memc > 0 and 'mem_reads' in o:
                lm = mem_consistency_loss(G, out, m, o, keep, a, ohw)
                lg = lg + a.lam_memc * lm
                parts['memc'] = lm.item()
            if a.lam_align > 0 and 'mem_reads' in o:
                la = mem_align_loss(G, x, m, o, keep)
                lg = lg + a.lam_align * la
                parts['align'] = la.item()
            if a.lam_speckle > 0:
                ls_ = speckle_loss(out, x, o['m_eff'])
                lg = lg + a.lam_speckle * ls_
                parts['speckle'] = ls_.item()
            if o_nb is not None:
                lt = temporal_loss(o, x, o_nb, b['nb'].to(dev), b['has_nb'].to(dev), a.tc_anchor)
                lg = lg + tc_w * lt
                parts['temp'] = lt.item()
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()

            parts['D'] = ld.item()
            if trk is not None:                                # 추적 prior 와 최종 anchor 거리 (prior 유효, px)
                vv = (b['has_pref'].to(dev) * tkeep) > 0.5
                if vv.any():
                    parts['trk_d'] = ((o['anchor'].detach().float() - trk)[vv] * x.shape[-1]).norm(dim=1).mean().item()
            if G2 is not None:                                 # ---- 역방향 G2 (vis2ani) 학습 step
                b2 = next(it2)
                x2, m2 = b2['x'].to(dev, non_blocking=True), b2['mask'].to(dev, non_blocking=True)
                ref2, has2 = b2['ref'].to(dev, non_blocking=True), b2['has_ref'].to(dev, non_blocking=True)
                xa2 = next(it_an2).to(dev, non_blocking=True)
                with torch.autocast(**amp):
                    o2 = G2(x2, m2)
                    o2_nb = G2(b2['nb'].to(dev, non_blocking=True), b2['nb_mask'].to(dev, non_blocking=True)) if tc_on else None
                out2 = o2['out'].float()
                with torch.autocast(**amp):
                    ld2 = crit_d.d_loss(D2(to_d(xa2, a)), D2(to_d(out2.detach(), a)))
                opt_d2.zero_grad(set_to_none=True); ld2.backward(); opt_d2.step()
                with torch.autocast(**amp):
                    df2 = D2(to_d(out2, a))
                    lgt2 = cls_logits(C, out2, a, ohw) if lam2['cls'] > 0 else None
                r2o, r2r = (out2[:, :1], ref2[:, :1]) if a.cls_input == 'rederive' else (out2, ref2)
                lg2, p2 = crit_g2(o2, df2, lgt2, r2o, r2r, has2, m2, lam2)
                if o2_nb is not None:
                    lt2 = temporal_loss(o2, x2, o2_nb, b2['nb'].to(dev), b2['has_nb'].to(dev), a.tc_anchor)
                    lg2 = lg2 + tc_w * lt2
                    p2['temp'] = lt2.item()
                opt_g2.zero_grad(set_to_none=True); lg2.backward(); opt_g2.step()
                p2['D'] = ld2.item()
                parts.update({'r_' + k: v for k, v in p2.items() if k in ('cls', 'gan', 'area', 'temp', 'D')})
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        sch_g.step(); sch_d.step()
        if G2 is not None:
            sch_g2.step(); sch_d2.step()
            res2 = evaluate(G2, C, dl2_va, a2, ohw, dev, save_png=os.path.join(a.out, f'r_val_ep{ep:03d}.png'))
            if a.adaptive and ep > a.warmup + a.ramp and res2['success'] < 0.9:
                state2['cls'] = min(a.cls_cap, state2['cls'] * 1.5)

        res = evaluate(G, C, dl_va, a, ohw, dev, save_png=os.path.join(a.out, f'val_ep{ep:03d}.png'), G2=G2)
        if G2 is not None:
            res.update({'r_' + k: v for k, v in res2.items() if k in ('success', 'pt', 'ps', 'area', 'anc_mx', 'anc_my', 'anc_sx', 'anc_sy')})
        if a.tgt_in:                                        # target 클래스 입력: P(tgt) 입력->출력, |Δ|, 경계 포함 flicker
            rt = evaluate(G, C, dl_vt, a, ohw, dev)
            res.update({'t_pt_in': rt['pv_in'] if tgt == 'visible' else rt['pa_in'], 't_pt': rt['pt'], 't_dmag': rt['dmag'],
                        't_area': rt['area'], 't_flick': rt.get('flicker', float('nan'))})
        tr = {k: v / max(n, 1) for k, v in agg.items()}
        tag = ''
        if ep > a.warmup:
            score = select_score(res, a)
            if score > best:
                best = score
                torch.save({'G': G.state_dict(), 'args': vars(a), 'epoch': ep, 'val': res,
                            **({'G2': G2.state_dict(), 'args2': vars(a2)} if G2 is not None else {})}, os.path.join(a.out, 'G_best.pt'))
                tag = ' *best'
                if a.sel_area or a.sel_dmag or a.sel_flick:
                    tag += f' (score={score:.3f})'
            if a.adaptive and ep > a.warmup + a.ramp and res['success'] < 0.9:
                state['cls'] = min(a.cls_cap, state['cls'] * 1.5)
        torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                    'args': vars(a), 'epoch': ep, 'state': state,
                    **({'G2': G2.state_dict(), 'D2': D2.state_dict(), 'opt_g2': opt_g2.state_dict(), 'opt_d2': opt_d2.state_dict(),
                        'state2': state2, 'args2': vars(a2)} if G2 is not None else {})}, os.path.join(a.out, 'last.pt'))
        print(f'[ep {ep:03d}] λcls={lam["cls"]:.2f} | ' + ' '.join(f'{k}={v:.4f}' for k, v in tr.items()) +
              f' || val succ={res["success"]:.3f} P(ani)={res["pa"]:.3f} P(vis)={res["pv"]:.3f} '
              f'P(abs)={res["pb"]:.3f} ' + (f'succ ref/noref={res["succ_ref"]:.3f}/{res["succ_noref"]:.3f} ' if 'succ_ref' in res else '') + f'area={res["area"]:.3f} |Δ|={res["dmag"]:.3f} flick={res.get("flicker", float("nan")):.4f} | ' +
              (f'tex β_hf={res["t_bhf"]:.3f} β_lf={res["t_blf"]:.3f} flow={res["t_flow"]:.1f}px | ' if 't_bhf' in res else '') +
              (f'anc μ=({res["anc_mx"]:.0f},{res["anc_my"]:.0f}) σ=({res["anc_sx"]:.1f},{res["anc_sy"]:.1f}) d1={res.get("anc_d1", float("nan")):.1f} | ' if 'anc_mx' in res else '') +
              (f'trk_d={res.get("trk_d", float("nan")):.1f} | rev succ={res["r_success"]:.3f} λ2={lam2["cls"]:.2f} anc σ=({res.get("r_anc_sx", float("nan")):.1f},{res.get("r_anc_sy", float("nan")):.1f}) | ' if 'r_success' in res else '') +
              (f'tgt P {res["t_pt_in"]:.3f}->{res["t_pt"]:.3f} area={res["t_area"]:.3f} |Δ|={res["t_dmag"]:.3f} flick={res["t_flick"]:.4f} | ' if a.tgt_in else '') +
              f'{time.time()-t0:.0f}s{tag}',
              flush=True)
        log.write(json.dumps({'epoch': ep, 'lam': lam, 'train': tr, 'val': res}) + '\n'); log.flush()
    print(f'done. best score={best:.4f} -> {a.out}/G_best.pt', flush=True)


if __name__ == '__main__':
    main()
