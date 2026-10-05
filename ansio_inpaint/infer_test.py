"""테스트셋 추론 + 저장.

각 비디오의 NERVE_VISIBLE 프레임에 Generator 적용 ->
  {out}/{vid}/{frame}.png : [입력 | 출력 | |Δ| jet] (원본 해상도, ΔI 를 원본 크기로 업샘플해 원본에 더함)
  {out}/{vid}.mp4         : 위 패널의 시간순 영상
  {out}/results.csv       : 프레임별 분류기 확률 (입력/출력)
Usage: python infer_test.py --ckpt runs/phys_v5/G_best_ep25.pt --out runs/phys_v5/test_infer
"""
import argparse
import csv
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from classifier import FrozenClassifier
from common import SEM2NAME, SIZE, load_input, original_hw, ref_flow, rederive, tissue_mask, to_tensor, videos
from data import _prior_mask, load_pvis, pvis_key
from runcfg import build_generator, resolve, src_tgt_probs


def put(img, text, y=22):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (4, y - h - 4), (12 + w, y + b), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--diff_max', type=float, default=0.3, help='jet 스케일 상한 (|Δ|, [0,1] 밝기 단위)')
    ap.add_argument('--fps', type=float, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--videos', nargs='*', default=None, help='지정 비디오 id 만 추론')
    ap.add_argument('--max_frames', type=int, default=0, help='>0: 비디오당 앞에서부터 최대 프레임 수')
    ap.add_argument('--full_video', action='store_true',
                    help='모든 클래스 프레임 포함(연속 영상). G 는 NERVE_VISIBLE 에만 적용, 나머지는 원본 그대로')
    ap.add_argument('--sam2g', action='store_true', help='SAM2G 스트리밍: SAM2 tracker 가 G 안에서 프레임마다 prior 생성 (--full_video 강제)')
    ap.add_argument('--no_track', action='store_true', help='체크포인트에 G2(추적용 역방향)가 있어도 추적 끄기')
    ap.add_argument('--anc_prior', type=float, default=0.0,
                    help='>0: 인과적 anchor 추적 (재학습 없음) — 직전 source 프레임 anchor 를 heatmap 에 Gaussian log-prior(σ, 이미지 비율)로 더함')
    ap.add_argument('--anc_prior_gap', type=int, default=5, help='anc_prior: 직전 anchor 를 쓰는 최대 프레임 간격')
    ap.add_argument('--track_sigma', type=float, default=None, help='추적 prior σ (기본: 학습값)')
    a = ap.parse_args()
    if a.sam2g:
        a.full_video = True
    dev = torch.device('cuda')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    t = SimpleNamespace(**ck['args'])
    from common import CLS_CKPT
    if getattr(t, 'size', 224) != SIZE or os.path.abspath(getattr(t, 'cls_ckpt', CLS_CKPT)) != os.path.abspath(CLS_CKPT):
        raise SystemExit(f'ckpt size={getattr(t, "size", 224)} cls={getattr(t, "cls_ckpt", None)} 인데 현재 SIZE={SIZE} CLS={CLS_CKPT}: '
                         f'ANSIO_SIZE / ANSIO_CLS_CKPT 환경변수를 맞추세요')
    cfg = resolve(t)
    G = build_generator(cfg).to(dev).eval()
    G.load_state_dict(ck['G'])
    G2 = None                                                                 # 추적: 프레임 순서대로 직전 anchor 를 다음 프레임 prior 로
    if 'G2' in ck and not a.no_track:                                         #   VIS 프레임 = G2(vis2ani) anchor 만 사용 (출력 이미지 미반영)
        G2 = build_generator(resolve(SimpleNamespace(**ck['args2']))).to(dev).eval()
        G2.load_state_dict(ck['G2'])
        trk_sigma = a.track_sigma or t.track_sigma
        trk_gap = t.track_gap
        a.full_video = True
        print(f'track: G2 {ck["args2"]["direction"]} σ={trk_sigma} gap≤{trk_gap}', flush=True)
    C = FrozenClassifier(repeat=t.repeat).to(dev)
    ohw = original_hw()
    tp, sp = cfg['target_p'], cfg['src_p']
    to_cls = (lambda z: rederive(z, ohw)) if t.cls_input == 'rederive' else (lambda z: z)
    os.makedirs(a.out, exist_ok=True)
    S2 = None
    if a.sam2g:
        from models.sam2g import SAM2G
        S2 = SAM2G(G, C, to_cls, SIZE, dev)
        trk_rows = []
    print(f'ckpt={a.ckpt} epoch={ck["epoch"]} direction={cfg["direction"]} '
          f'success: P({cfg["tgt_sem"]})>={tp} & P({cfg["src_sem"]})<={sp}')

    vis = SEM2NAME[cfg['src_sem']]                                           # 생성 대상(source) 클래스 폴더명
    tgt = SEM2NAME[cfg['tgt_sem']]
    mem_k = getattr(t, 'mem_k', 0) or 0                                     # 과거 feature 메모리: 앞선 mem_gap 프레임 중 분류기 P(vis)>=mem_thr 최근 K개
    use_ref = bool(getattr(t, 'ref_in', 0)) and not mem_k                                  # 과거 참조: 현재보다 앞선 가장 최근 GT target 프레임(≤ref_gap)
    rows, n_ok, n_all = [], 0, 0
    s2m = bool(getattr(t, 's2m', 0))                                        # train_s2m 체크포인트: 프레임 순서 스트리밍 + 자기 출력 메모리
    vids = sorted(videos(a.split, list(SEM2NAME.values()) if a.full_video else ([vis, tgt] if s2m else [vis])).items())
    if a.videos:
        vids = [(v, f) for v, f in vids if v in a.videos]
    if a.max_frames > 0:
        vids = [(v, f[:a.max_frames]) for v, f in vids]
    for vid, frames in vids:
        t_fr = [(i, p) for i, p, c in videos(a.split, [tgt]).get(vid, [])]
        t_idx = np.array([i for i, _ in t_fr])

        def past_ref(i):
            if not len(t_idx):
                return None
            k = np.searchsorted(t_idx, i) - 1
            if k < 0 or i - int(t_idx[k]) > t.ref_gap:
                return None
            return t_fr[k][1], i - int(t_idx[k])
        if mem_k:
            pvc = load_pvis()
            ok_fr = [(i, p) for i, p, _ in videos(a.split, list(SEM2NAME.values())).get(vid, []) if pvc[pvis_key(p)][0] >= t.mem_thr]

            def past_mem(i):
                return [(q, i - j) for j, q in reversed(ok_fr) if 0 < i - j <= t.mem_gap][:mem_k]
        vdir = os.path.join(a.out, vid)
        os.makedirs(vdir, exist_ok=True)
        writer = None
        if S2 is not None:
            S2.reset()
        bank = []
        trk_ref = None                                                        # (anchor [1,2], frame idx)
        anc_prev = None                                                       # anc_prior: (anchor [1,2], frame idx)
        for s in range(0, len(frames), a.batch):
            chunk = frames[s:s + a.batch]
            xs, ms, grays = [], [], []
            for _, p, _ in chunk:
                im, _ = load_input(p)
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                xs.append(to_tensor(im)); grays.append(g)
                ms.append(torch.from_numpy(_prior_mask(p, g, getattr(t, 'mask_dir', None), a.split))[None].float())   # mask_dir(SAM2 track) 있으면 사용
            x, m = torch.stack(xs).to(dev), torch.stack(ms).to(dev)
            rk = {}
            if use_ref:
                rs, rms, rv, rg = [], [], [], []
                for idx, _, _ in chunk:
                    pr = past_ref(idx)
                    if pr is None:
                        rs.append(torch.zeros_like(xs[0])); rms.append(torch.zeros_like(ms[0])); rv.append(0.); rg.append(0.)
                    else:
                        gr = cv2.imread(pr[0], cv2.IMREAD_GRAYSCALE)
                        rs.append(to_tensor(load_input(pr[0])[0])); rms.append(torch.from_numpy(tissue_mask(gr))[None].float())
                        rv.append(1.); rg.append(float(pr[1]))
                rk = dict(ref=torch.stack(rs).to(dev), ref_m=torch.stack(rms).to(dev),
                          ref_v=torch.tensor(rv, device=dev), ref_gap=torch.tensor(rg, device=dev))
                if getattr(t, 'tex_fixed', 0):                                # 고정 정합 flow (DIS, 참조 -> 현재)
                    rk['ref_flow'] = torch.stack([ref_flow(xs[i], rs[i]) if rv[i] else torch.zeros(2, *xs[0].shape[-2:])
                                                  for i in range(len(chunk))]).to(dev)
            if mem_k:
                K = mem_k
                mx = torch.zeros(len(chunk), K, *xs[0].shape); mm = torch.zeros(len(chunk), K, *ms[0].shape)
                mv, mg = torch.zeros(len(chunk), K), torch.zeros(len(chunk), K)
                for bi, (idx, _, _) in enumerate(chunk):
                    for s_, (q, gap) in enumerate(past_mem(idx)):
                        gq = cv2.imread(q, cv2.IMREAD_GRAYSCALE)
                        mx[bi, s_] = to_tensor(load_input(q)[0]); mm[bi, s_] = torch.from_numpy(tissue_mask(gq))[None].float()
                        mv[bi, s_], mg[bi, s_] = 1., float(gap)
                rk = dict(mem=mx.to(dev), mem_m=mm.to(dev), mem_v=mv.to(dev), mem_gap=mg.to(dev))
            if S2 is not None:                                                    # 프레임 순서대로 SAM2 tracking -> prior -> G
                os_, ps_ = [], []
                for i, (idx, p, cname) in enumerate(chunk):
                    oi, pri, st, pvi, bx = S2.step(s + i, grays[i], x[i:i + 1])
                    os_.append(oi); ps_.append(torch.from_numpy(pri)[None])
                    trk_rows.append([vid, idx, cname, st, f'{pvi:.4f}', '' if bx is None else ' '.join(f'{v:.1f}' for v in bx)])
                o = {k: torch.cat([q[k] for q in os_]) for k in ('out', 'm_eff')}
                m = torch.stack(ps_).to(dev).float()
            elif s2m:                                                             # stream_eval 과 동일: source/target 프레임 모두 G + 메모리 갱신
                os_ = []
                for i in range(len(chunk)):
                    with torch.autocast('cuda', torch.bfloat16):
                        oi = G(x[i:i + 1], m[i:i + 1], s2_bank=bank)
                        f_, p_ = G.s2m.write(oi['s2_pre'], oi['m_eff'])
                    bank = ([(bf, bp, tr + 1) for bf, bp, tr in bank] + [(f_, p_, torch.ones(1, device=dev))])[-(G.s2m.num_maskmem - 1):]
                    os_.append(oi)
                o = {k: torch.cat([q[k].float() for q in os_]) for k in ('out', 'm_eff')}
            elif a.anc_prior > 0:                                                 # 인과적 anchor 추적: 프레임 순차, 직전 anchor = prior
                outs, an_l, an1_l, srcs = [], [], [], []
                for i, (idx, _, cname) in enumerate(chunk):
                    if not (cname == vis or (getattr(t, 'tgt_in', 0) and cname == tgt)):   # 비대상: 원본 유지 (추적 상태는 gap 으로 만료)
                        outs.append(x[i:i + 1].float()); an_l.append(None); an1_l.append(None); srcs.append('')
                        continue
                    ok_p = anc_prev is not None and idx - anc_prev[1] <= a.anc_prior_gap
                    G.anc_ref = anc_prev[0] if ok_p else None
                    G.anc_ref_valid, G.anc_ref_sigma = torch.ones(1, device=dev), a.anc_prior
                    with torch.no_grad():
                        oi = G(x[i:i + 1], m[i:i + 1], **{k: v[i:i + 1] for k, v in rk.items()})
                    G.anc_ref = None
                    anc_prev = (oi['anchor'].float(), idx)
                    outs.append(oi['out'].float()); an_l.append(oi['anchor'][0].float().cpu().numpy())
                    an1_l.append(oi['anchor1'][0].float().cpu().numpy() if oi.get('anchor1') is not None else None)
                    srcs.append('a2v+prior' if ok_p else 'a2v')
                o = {'out': torch.cat(outs)}
            elif G2 is not None:                                                  # 추적: 프레임 순차
                outs, an_l, an1_l, srcs = [], [], [], []
                for i, (idx, _, cname) in enumerate(chunk):
                    ok_ref = trk_ref is not None and idx - trk_ref[1] <= trk_gap
                    net = G if cname == vis else (G2 if cname == tgt else None)
                    if net is None:                                               # absent: 원본 유지, 추적 상태 유지
                        outs.append(x[i:i + 1].float()); an_l.append(None); an1_l.append(None); srcs.append('')
                        continue
                    net.anc_ref = trk_ref[0] if ok_ref else None
                    net.anc_ref_valid, net.anc_ref_sigma = torch.ones(1, device=dev), trk_sigma
                    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                        oi = net(x[i:i + 1], m[i:i + 1])
                    net.anc_ref = None
                    an_ = oi['anchor'].float()
                    trk_ref = (an_, idx)
                    an_l.append(an_[0].cpu().numpy())
                    an1_l.append(oi['anchor1'][0].float().cpu().numpy() if oi.get('anchor1') is not None else None)
                    srcs.append(('a2v' if net is G else 'v2a') + ('+ref' if ok_ref else ''))
                    outs.append(oi['out'].float() if net is G else x[i:i + 1].float())   # G2 출력은 버림
                o = {'out': torch.cat(outs)}
            else:
                with torch.no_grad():
                    o = G(x, m, **rk)
            pv0, pa0, pb0 = C.sem_probs(C(to_cls(x)))
            ps, pt, pv, pa, pb = src_tgt_probs(C, C(to_cls(o['out'].float())), cfg)
            d224 = ((o['out'][:, 0].float() - x[:, 0]) * 0.5).cpu().numpy()        # ΔI in [0,1] 밝기 단위
            if G2 is not None or a.anc_prior > 0:                                 # 프레임별 (최종, pass1) anchor 또는 None
                fa = list(zip(an_l, an1_l))
            else:
                an = o['anchor'].float().cpu().numpy() if o.get('anchor') is not None else None     # 최종 창 anchor (x, y) [0,1]
                an1 = o['anchor1'].float().cpu().numpy() if o.get('anchor1') is not None else None  # pass1 (SCA prior) anchor
                srcs = ['a2v' if c == vis else '' for _, _, c in chunk]
                fa = [((an[i] if an is not None else None), (an1[i] if an1 is not None else None)) if c == vis else (None, None)
                      for i, (_, _, c) in enumerate(chunk)]
            for i, (idx, p, cname) in enumerate(chunk):
                is_vis = cname == vis
                edit = is_vis or (getattr(t, 'tgt_in', 0) and cname == tgt)        # tgt_in 학습본: target 클래스 프레임도 G 적용
                if not edit:                                                     # 비대상 프레임: 원본 유지
                    d224[i] = 0
                g = grays[i]
                H, W = g.shape
                d = cv2.resize(d224[i], (W, H), interpolation=cv2.INTER_LINEAR)
                out = np.clip(g.astype(np.float32) / 255 + d, 0, 1)
                out_u8 = (out * 255).round().astype(np.uint8)
                jet = cv2.applyColorMap((np.clip(np.abs(d) / a.diff_max, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                gin, gout = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)
                ok = bool(is_vis and pt[i] >= tp and ps[i] <= sp)
                put(gin, f'input [{cname}] P(vis)={pv0[i]:.2f} P(ani)={pa0[i]:.2f}')
                if edit:
                    put(gout, f'output P(vis)={pv[i]:.2f} P(ani)={pa[i]:.2f} {"OK" if ok else ""}')
                else:
                    put(gout, f'unchanged (GT {cname})')
                    pv[i], pa[i], pb[i] = pv0[i], pa0[i], pb0[i]
                put(jet, f'|diff| jet 0-{a.diff_max}')
                panel = np.concatenate([gin, gout, jet], 1)
                put(panel, f'{vid} f{idx:05d}', y=H - 10)
                cv2.imwrite(os.path.join(vdir, f'{os.path.splitext(os.path.basename(p))[0]}.png'), panel)
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(a.out, f'{vid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                             a.fps, (panel.shape[1], panel.shape[0]))
                writer.write(panel)
                rows.append([vid, idx, os.path.basename(p), cname, int(edit), f'{pv0[i]:.4f}', f'{pa0[i]:.4f}', f'{pb0[i]:.4f}',
                             f'{pv[i]:.4f}', f'{pa[i]:.4f}', f'{pb[i]:.4f}', int(ok), f'{np.abs(d).mean():.4f}'] +
                            [f'{v:.4f}' if q is not None else '' for q in fa[i] for v in (q if q is not None else (0, 0))] + [srcs[i]])
                n_ok += ok; n_all += is_vis
        if writer is not None:
            writer.release()
        vr = [r for r in rows if r[0] == vid]
        nv = sum(r[3] == vis for r in vr)                                          # 성공률은 source 프레임 기준
        print(f'  {vid}: frames={len(vr)} visible(generated)={nv} success={sum(r[11] for r in vr) / max(nv, 1):.3f}', flush=True)
    if S2 is not None:
        with open(os.path.join(a.out, 'sam2g_track.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['video', 'frame', 'gt', 'prior', 'pvis_in', 'prompt_box']); w.writerows(trk_rows)
        from collections import Counter
        print('sam2g prior source:', dict(Counter(r[3] for r in trk_rows)), 'prompts:', sum(r[5] != '' for r in trk_rows))
    with open(os.path.join(a.out, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'frame', 'file', 'gt', 'generated', 'in_P_vis', 'in_P_ani', 'in_P_abs', 'out_P_vis', 'out_P_ani', 'out_P_abs',
                    'success', 'mean_abs_diff', 'anc_x', 'anc_y', 'anc1_x', 'anc1_y', 'anc_src'])
        w.writerows(rows)
    print(f'[{a.split}] generated n={n_all} success={n_ok / max(n_all, 1):.3f} -> {a.out}')


if __name__ == '__main__':
    main()
