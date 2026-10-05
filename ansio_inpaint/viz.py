"""infer_test.py 결과 폴더 재시각화 (GPU 불필요: 패널 PNG + results.csv 만 사용).

프레임 레이아웃
  [입력 | 출력 | 변화 overlay]  +  하단 타임라인
  - 입력/출력 위: 클래스 확률 스택 바 (vis=파랑 | absent=회색 | ani=주황).
    출력 바에는 성공 기준 눈금 (P(vis)>=target_p, P(ani)<=src_p) 표시, 성공 시 출력 패널 녹색 테두리.
  - overlay: 출력(어둡게) 위에 부호 있는 ΔI 를 연속 alpha 로 덧칠 (밝아짐=노랑, 어두워짐=보라).
    하드 threshold 없음. 스케일은 비디오 단위로 고정(|Δ| 99.5 퍼센타일) -> 프레임 간 깜빡임 없음.
  - 타임라인: 비디오 전체의 출력 P(vis)/P(ani), 입력 P(vis)(회색), target 선, 성공 프레임 틱, 현재 위치 커서.
    GT 리본(vis=파랑 | absent=빗금 회색 | ani=주황), GT absent 구간은 플롯 배경도 빗금 음영.
Usage: python viz.py runs/a2v_sca_v4/test_infer_450_best_ep46 [--out DIR] [--fps 10]
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def bgr(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


C_VIS, C_ANI, C_ABS = bgr('#3987e5'), bgr('#d95926'), bgr('#6f6e69')
C_UP, C_DN = np.array(bgr('#fab219'), np.float32), np.array(bgr('#9085e9'), np.float32)
C_OK, C_TXT, C_MUTED, C_GRID = bgr('#0ca30c'), (255, 255, 255), bgr('#c3c2b7'), bgr('#383835')
C_ANC, C_ANC1, C_TRK = bgr('#ff3b3b'), bgr('#4be04b'), bgr('#2ad4e0')
TRAIL = 15          # anchor 궤적: 최근 N 프레임
C_GT = {'NERVE_VISIBLE': C_VIS, 'ANISOTROPIC': C_ANI, 'NERVE_NOT_VISIBLE': C_ABS}
FONT = cv2.FONT_HERSHEY_SIMPLEX
TOP = 64            # 패널 상단 헤더 높이 (원본 패널 여백 위에 덮어씀)
TL_H = 112          # 타임라인 높이
GAMMA = 1.5        # >1: 미세한 전역 변화는 옅게, 큰 변화만 진하게 (하드 threshold 대신)


def text(img, s, x, y, scale=0.5, color=C_TXT, th=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, th, cv2.LINE_AA)


def prob_bar(img, x, y, w, h, pv, pa, pb, tp=None, sp=None):
    """vis 는 왼쪽부터, ani 는 오른쪽부터, absent 는 가운데. 세그먼트 사이 2px 간격."""
    xv, xa = x + int(round(w * pv)), x + w - int(round(w * pa))
    for x0, x1, c in ((x, xv, C_VIS), (xv, xa, C_ABS), (xa, x + w, C_ANI)):
        if x1 - x0 > 2:
            cv2.rectangle(img, (x0 + 1, y), (x1 - 1, y + h), c, -1)
    for x0, x1, name, p in ((x, xv, 'vis', pv), (xa, x + w, 'ani', pa)):
        s = f'{name} {p:.2f}'
        tw = cv2.getTextSize(s, FONT, 0.42, 1)[0][0]
        if x1 - x0 > tw + 8:
            text(img, s, (x0 + 5) if name == 'vis' else (x1 - tw - 5), y + h - 5, 0.42)
    if tp is not None:                                                # 성공 기준 눈금
        for t in (tp, 1 - sp):
            xt = x + int(round(w * t))
            cv2.line(img, (xt, y - 4), (xt, y + h + 4), C_TXT, 2)


def overlay(g, d, scale, sigma=1.5):
    """출력 grayscale 을 어둡게 깔고 ΔI 부호별 색을 alpha=(|Δ|/scale)^GAMMA 로 블렌딩."""
    ds = cv2.GaussianBlur(d, (0, 0), sigma)
    a = np.clip(np.abs(ds) / scale, 0, 1) ** GAMMA * 0.9
    base = np.repeat((g.astype(np.float32) * 0.55)[..., None], 3, 2)
    col = np.where((ds >= 0)[..., None], C_UP, C_DN)
    return (base * (1 - a[..., None]) + col * a[..., None]).clip(0, 255).astype(np.uint8)


def legend(img, x, y, w, scale):
    h = 10
    for i in range(w):
        v = (i / (w - 1)) * 2 - 1                                     # -1..1
        c = C_UP if v >= 0 else C_DN
        a = abs(v) ** GAMMA * 0.9
        cv2.line(img, (x + i, y), (x + i, y + h), tuple(int(k) for k in c * a), 1)
    text(img, f'-{scale:.2f}', x, y + h + 16, 0.4, C_MUTED)
    text(img, '0', x + w // 2 - 4, y + h + 16, 0.4, C_MUTED)
    text(img, f'+{scale:.2f}', x + w - 40, y + h + 16, 0.4, C_MUTED)
    text(img, 'darker', x, y - 6, 0.4, C_MUTED)
    text(img, 'brighter', x + w - 62, y - 6, 0.4, C_MUTED)


def timeline(W, rows, cur, tp):
    tl = np.zeros((TL_H, W, 3), np.uint8)
    n = len(rows)
    L, R, T, B = 60, W - 12, 10, TL_H - 38
    X = lambda i: int(L + (R - L) * i / max(n - 1, 1))
    Y = lambda p: int(B - (B - T) * p)
    for i, r in enumerate(rows):                                      # 비생성(원본 유지) 구간 음영
        if not int(r['generated']):
            cv2.line(tl, (X(i), T), (X(i), B), (30, 30, 30), max(1, (R - L) // max(n, 1) + 1))
    hw = max(1, (R - L) // max(n, 1) + 1)
    hatch = ((np.add.outer(np.arange(TL_H), np.arange(W)) // 4) % 2 == 0)       # 대각 빗금 마스크
    for i, r in enumerate(rows):                                      # GT absent 구간: 플롯 배경 빗금
        if r['gt'] == 'NERVE_NOT_VISIBLE':
            x0, x1 = X(i) - hw // 2, X(i) + hw - hw // 2
            sub = tl[T:B, x0:x1]
            sub[hatch[T:B, x0:x1]] = (70, 64, 64)
    for i, r in enumerate(rows):                                      # GT 리본
        x0, x1 = X(i) - hw // 2, X(i) + hw - hw // 2
        tl[B + 14:B + 24, x0:x1] = C_GT.get(r['gt'], C_GRID)
        if r['gt'] == 'NERVE_NOT_VISIBLE':
            sub = tl[B + 14:B + 24, x0:x1]
            sub[hatch[B + 14:B + 24, x0:x1]] = (40, 40, 40)
    text(tl, 'GT', 30, B + 23, 0.38, C_MUTED)
    for p in (0, 0.5, 1):
        cv2.line(tl, (L, Y(p)), (R, Y(p)), C_GRID, 1)
        text(tl, f'{p:.1f}', 22, Y(p) + 4, 0.38, C_MUTED)
    cv2.line(tl, (L, Y(tp)), (R, Y(tp)), C_MUTED, 1, cv2.LINE_AA)
    for key, c, th in (('in_P_vis', C_ABS, 1), ('out_P_ani', C_ANI, 2), ('out_P_vis', C_VIS, 2)):
        pts = np.array([[X(i), Y(float(r[key]))] for i, r in enumerate(rows)], np.int32)
        cv2.polylines(tl, [pts], False, c, th, cv2.LINE_AA)
    for i, r in enumerate(rows):
        if int(r['success']):
            cv2.line(tl, (X(i), B + 4), (X(i), B + 10), C_OK, 1)
    cv2.line(tl, (X(cur), T - 4), (X(cur), B + 10), C_TXT, 1)
    x = L
    for s, c in (('out P(vis)', C_VIS), ('out P(ani)', C_ANI), ('in P(vis)', C_ABS), ('success', C_OK),
                 ('GT absent (hatched)', (70, 64, 64))):
        cv2.line(tl, (x, TL_H - 7), (x + 14, TL_H - 7), c, 2)
        text(tl, s, x + 18, TL_H - 3, 0.38, C_MUTED)
        x += 30 + cv2.getTextSize(s, FONT, 0.38, 1)[0][0]
    text(tl, f'target {tp}', R - 70, Y(tp) - 4, 0.38, C_MUTED)
    return tl


def draw_anchor(img, rows, i, W, H, r_):
    """빨강 점 + 타원 = 최종 창 anchor (반경 = anchor_r σ), 초록 + = pass1 anchor, 옅은 선 = 최근 TRAIL 프레임 궤적."""
    get = lambda r, k: (float(r[k + '_x']) * W, float(r[k + '_y']) * H) if r.get(k + '_x') else None
    pts = [get(r, 'anc') for r in rows[max(0, i - TRAIL):i + 1]]
    pts = [p for p in pts if p is not None]
    for j in range(1, len(pts)):
        cv2.line(img, tuple(map(int, pts[j - 1])), tuple(map(int, pts[j])), C_ANC, 1, cv2.LINE_AA)
    c, c1 = get(rows[i], 'anc'), get(rows[i], 'anc1')
    if c1 is not None:
        cv2.drawMarker(img, tuple(map(int, c1)), C_ANC1, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    if c is not None:
        cv2.ellipse(img, tuple(map(int, c)), (int(r_ * W), int(r_ * H)), 0, 0, 360, C_ANC, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, c)), 4, C_ANC, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--out', default=None, help='기본: {dir}_viz')
    ap.add_argument('--fps', type=float, default=10)
    ap.add_argument('--scale', type=float, default=0, help='>0: overlay |Δ| 스케일 고정 (기본: 비디오별 99.5 퍼센타일)')
    ap.add_argument('--target_p', type=float, default=None)
    ap.add_argument('--src_p', type=float, default=None)
    a = ap.parse_args()
    out_root = a.out or a.dir.rstrip('/') + '_viz'
    run_args = os.path.join(os.path.dirname(a.dir.rstrip('/')), 'args.json')
    cfg = json.load(open(run_args)) if os.path.exists(run_args) else {}
    tp = a.target_p if a.target_p is not None else cfg.get('target_p', 0.6)
    sp = a.src_p if a.src_p is not None else cfg.get('src_p', 0.2)
    anc_r = cfg.get('anchor_r', 0.0)

    by_vid = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.dir, 'results.csv'))):
        by_vid[r['video']].append(r)
    for vid, rows in sorted(by_vid.items()):
        rows.sort(key=lambda r: int(r['frame']))
        panels = [cv2.imread(os.path.join(a.dir, vid, r['file'].replace('.png', '') + '.png'), cv2.IMREAD_GRAYSCALE)
                  for r in rows]
        H, W3 = panels[0].shape
        W = W3 // 3
        diffs = []
        for p in panels:
            d = p[:, W:2 * W].astype(np.float32) / 255 - p[:, :W].astype(np.float32) / 255
            d[:TOP], d[-40:] = 0, 0                                     # 텍스트 오버레이 영역 제외
            diffs.append(d)
        nz = np.concatenate([np.abs(d[d != 0]) for d in diffs] + [np.zeros(1, np.float32)])
        scale = a.scale or max(float(np.percentile(nz, 99.5)), 0.02)
        os.makedirs(os.path.join(out_root, vid), exist_ok=True)
        writer = None
        for i, (r, p, d) in enumerate(zip(rows, panels, diffs)):
            gin, gout = p[:, :W].copy(), p[:, W:2 * W].copy()
            gin[:TOP], gout[:TOP] = 0, 0
            gin[-40:], gout[-40:] = 0, 0
            gen, ok = int(r['generated']), int(r['success'])
            pin = cv2.cvtColor(gin, cv2.COLOR_GRAY2BGR)
            pout = cv2.cvtColor(gout, cv2.COLOR_GRAY2BGR)
            pov = overlay(gout, d, scale) if gen else cv2.cvtColor((gout * 0.55).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            f = lambda k: float(r[k])
            gt = r['gt'].replace('NERVE_', '')
            text(pin, f'input  (GT {gt})', 8, 20)
            prob_bar(pin, 8, 32, W - 16, 22, f('in_P_vis'), f('in_P_ani'), f('in_P_abs'))
            if gen:
                text(pout, 'output', 8, 20)
                prob_bar(pout, 8, 32, W - 16, 22, f('out_P_vis'), f('out_P_ani'), f('out_P_abs'), tp, sp)
                if ok:
                    cv2.rectangle(pout, (1, 1), (W - 2, H - 2), C_OK, 3)
                    text(pout, 'OK', W - 38, 20, 0.55, C_OK, 2)
                text(pov, 'change  (output - input)', 8, 20)
                legend(pov, W - 190, H - 30, 180, scale)
            else:
                text(pout, 'unchanged (not a source frame)', 8, 20, 0.5, C_MUTED)
                text(pov, 'no change', 8, 20, 0.5, C_MUTED)
            text(pin, f'{vid}  f{int(r["frame"]):05d}', 8, H - 12, 0.45, C_MUTED)
            if gen and r.get('anc_x'):                                  # anchor 위치 (입력/출력/변화 패널 모두)
                for im_ in (pin, pout, pov):
                    draw_anchor(im_, rows, i, W, H, anc_r)
                text(pout, 'anchor: red=final(window r)  green+=pass1  line=trail', 8, H - 12, 0.42, C_MUTED)
            elif r.get('anc_x'):                                        # 추적 전용 anchor (VIS 프레임 G2): 입력 패널에 점만, 변환 결과는 미반영
                c = (int(float(r['anc_x']) * W), int(float(r['anc_y']) * H))
                draw_anchor(pin, rows, i, W, H, 0)
                cv2.circle(pin, c, 5, C_TRK, -1, cv2.LINE_AA)
                text(pin, 'track anchor (vis2ani, image not used)', 8, H - 30, 0.42, C_TRK)
            frame = np.concatenate([np.concatenate([pin, pout, pov], 1), timeline(3 * W, rows, i, tp)], 0)
            cv2.imwrite(os.path.join(out_root, vid, r['file']), frame)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(out_root, f'{vid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         a.fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        writer.release()
        print(f'  {vid}: frames={len(rows)} scale={scale:.3f}', flush=True)
    print(f'-> {out_root}')


if __name__ == '__main__':
    main()
