"""데이터셋.

VisibleDS : Nerve Visible 프레임 I_vis + prior mask M + (선택) pseudo-paired anisotropic 참조 I_ref
            I_ref = 같은 비디오에서 시간상 가장 가까운 ANISOTROPIC 프레임(|Δframe| <= max_gap 일 때만).
            인접 프레임이라 프로브 위치가 거의 같아 L_rec 의 약한 paired 감독으로 쓸 수 있다.
AnisoDS   : 판별기용 실제 ANISOTROPIC 프레임.
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import SEM2NAME, SIZE, load_input, tissue_mask, to_tensor, videos, ref_flow

PVIS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pvis_cache.json')
_PV = None


def load_pvis():
    """cache_pvis.py 결과: {real_temporary 기준 상대경로: [P(vis), P(ani), P(abs)]}"""
    global _PV
    if _PV is None:
        import json
        _PV = json.load(open(PVIS_CACHE))
    return _PV


def pvis_key(p):
    return os.path.relpath(os.path.abspath(p), os.path.dirname(os.path.dirname(PVIS_CACHE)))


def _anc_target(gpath, anc_dir, max_area=0.25):
    """{anc_dir}/{split}/{CLASS}/{name} 신경 track 마스크(make_track_masks) 무게중심 (x, y) in [0,1].
    마스크 없음 / 면적 > max_area (조직 prior fallback) 이면 None."""
    rel = os.path.relpath(os.path.abspath(gpath), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(gpath)))))
    q = os.path.join(anc_dir, rel)
    if not os.path.exists(q):
        return None
    mm = cv2.imread(q, cv2.IMREAD_GRAYSCALE) > 127
    if not mm.any() or mm.mean() > max_area:
        return None
    ys, xs = np.nonzero(mm)
    return (xs.mean() + 0.5) / mm.shape[1], (ys.mean() + 0.5) / mm.shape[0]


def _prior_mask(gpath, g_orig, mask_dir, split):
    if mask_dir:                                           # 외부 마스크: {mask_dir}/{split}/{CLASS}/{name}.png
        rel = os.path.relpath(os.path.abspath(gpath), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(gpath)))))
        p = os.path.join(mask_dir, rel)                    # {split}/{CLASS}/{name}: split 은 경로에서 ('train+val' 대응)
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            return cv2.resize(m, (SIZE, SIZE))
    return tissue_mask(g_orig)


class SourceDS(Dataset):
    """source 클래스 프레임 + prior mask + 같은 비디오의 가장 가까운 ref(target) 클래스 프레임(pseudo-pair)."""
    def __init__(self, split, train=False, max_gap=5, mask_dir=None, src='visible', ref='anisotropic', nb_gap=0, pref_gap=0,
                 mem_k=0, mem_gap=30, mem_thr=0.5, tgt_in=False, anc_dir=None, ref_flow=False):
        """nb_gap > 0: 같은 비디오의 인접 source 프레임(0 < |Δframe| <= nb_gap)을 'nb' 로 함께 반환 (temporal loss 용).
        train: 후보 중 랜덤, eval: 가장 가까운 다음(없으면 이전) 프레임 -> 결정적."""
        """pref_gap > 0: 과거 참조 — 같은 비디오에서 현재보다 앞선(과거) 가장 가까운 ref 클래스 프레임(0 < i-j <= pref_gap).
        'pref'/'pref_mask'/'has_pref'/'pref_gap' 로 반환 (nb 프레임도 자기 과거 참조를 'nb_pref*' 로)."""
        """mem_k > 0: 과거 feature 메모리 — 같은 비디오에서 현재보다 앞선(0 < i-j <= mem_gap) 프레임 중
        분류기 P(vis)(원본 입력, pvis_cache.json) >= mem_thr 인 가장 최근 mem_k 개 (GT 라벨 미사용 -> 실제 추론과 동일 조건).
        'mem' [K,3,H,W] / 'mem_mask' [K,1,H,W] / 'mem_v' [K] / 'mem_gap' [K] 로 반환 (nb 프레임도 'nb_mem*')."""
        """tgt_in: target 클래스 프레임(이미 목표 상태, 예: ani2vis 의 VISIBLE)도 G 입력 item 으로 추가(is_src=0, ref 없음).
        이때 temporal 이웃은 source/target 클래스를 가리지 않음(경계를 넘는 쌍 포함)."""
        self.split, self.train, self.mask_dir, self.nb_gap, self.pref_gap = split, train, mask_dir, nb_gap, pref_gap
        self.ref_flow = ref_flow                            # 과거 참조 -> 현재 고정 정합 flow (DIS, flip 후 계산) 'pref_flow'
        self.mem_k, self.anc_dir = mem_k, anc_dir
        self.mem_of = {}
        if mem_k:
            pv = load_pvis()
            for vid, fr in videos(split, list(SEM2NAME.values())).items():
                ok = [(i, p) for i, p, _ in fr if pv[pvis_key(p)][0] >= mem_thr]
                for i, p, _ in fr:
                    self.mem_of[p] = [(q, i - j) for j, q in reversed(ok) if 0 < i - j <= mem_gap][:mem_k]
        vis, ani = SEM2NAME[src], SEM2NAME[ref]
        self.is_src = []                                    # 1: source 클래스 item, 0: target 클래스 item (tgt_in)
        self.items = []                                     # (src_path, ref_path|None, [(nb_path, past_ref|None)], past_ref|None)
        for vid, frames in videos(split, [vis, ani]).items():
            a_idx = np.array([i for i, _, c in frames if c == ani])
            a_path = [p for _, p, c in frames if c == ani]
            s_fr = [(i, p) for i, p, c in frames if c == vis]
            nb_fr = [(i, p) for i, p, _ in frames] if tgt_in else s_fr   # tgt_in: 클래스 경계를 넘는 이웃 허용

            def past_ref(i):                                # 인과적(과거만) 참조: (path, gap) or None
                if not pref_gap or not len(a_idx):
                    return None
                k = np.searchsorted(a_idx, i) - 1           # a_idx 는 시간순 -> i 보다 작은 마지막 인덱스
                if k < 0 or i - int(a_idx[k]) > pref_gap:
                    return None
                return a_path[k], i - int(a_idx[k])
            for i, p, c in frames:
                if c != vis:
                    if tgt_in:                              # target 클래스 프레임: ref/pseudo-pair 없음
                        nbs = sorted(((abs(k - i), -(k - i), k, q) for k, q in nb_fr if 0 < abs(k - i) <= nb_gap)) if nb_gap else []
                        self.items.append((p, None, [(q, past_ref(k)) for *_, k, q in nbs], past_ref(i)))
                        self.is_src.append(0)
                    continue
                ref = None
                if len(a_idx):
                    j = int(np.argmin(np.abs(a_idx - i)))
                    if abs(int(a_idx[j]) - i) <= max_gap:
                        ref = a_path[j]
                nbs = sorted(((abs(k - i), -(k - i), k, q) for k, q in nb_fr if 0 < abs(k - i) <= nb_gap)) if nb_gap else []
                self.items.append((p, ref, [(q, past_ref(k)) for *_, k, q in nbs], past_ref(i)))
                self.is_src.append(1)
        self.n_paired = sum(it[1] is not None for it in self.items)
        self.n_nb = sum(len(it[2]) > 0 for it in self.items)
        self.n_pref = sum(it[3] is not None for it in self.items)

    def __len__(self):
        return len(self.items)

    def _load(self, p):
        im, _ = load_input(p)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return to_tensor(im), torch.from_numpy(_prior_mask(p, g, self.mask_dir, self.split))[None].float()

    def _pref(self, pr, like_x, like_m):
        if pr is None:
            return torch.zeros_like(like_x), torch.zeros_like(like_m), torch.tensor(0.0), torch.tensor(0.0)
        r, rm = self._load(pr[0])
        return r, rm, torch.tensor(1.0), torch.tensor(float(pr[1]))

    def _mem(self, p, like_x, like_m):
        K = self.mem_k
        ms = self.mem_of.get(p, []) if p is not None else []
        xs, mm = like_x.new_zeros(K, *like_x.shape), like_m.new_zeros(K, *like_m.shape)
        v, g = torch.zeros(K), torch.zeros(K)
        for s, (q, gap) in enumerate(ms):
            xs[s], mm[s] = self._load(q)
            v[s], g[s] = 1.0, float(gap)
        return xs, mm, v, g

    def __getitem__(self, k):
        p, ref, nbs, pref = self.items[k]
        x, m = self._load(p)
        if ref is not None:
            r, has = to_tensor(load_input(ref)[0]), torch.tensor(1.0)
        else:
            r, has = torch.zeros_like(x), torch.tensor(0.0)
        out = {'path': p, 'is_src': torch.tensor(float(self.is_src[k]))}
        if self.nb_gap:
            if nbs:
                qn, nb_pref = random.choice(nbs) if self.train else nbs[0]
                xn, mn = self._load(qn)
                hn = torch.tensor(1.0)
            else:
                xn, mn, hn, nb_pref = torch.zeros_like(x), torch.zeros_like(m), torch.tensor(0.0), None
                qn = None
        if self.pref_gap:
            pr, prm, hp, gp = self._pref(pref, x, m)
            if self.nb_gap:
                npr, nprm, nhp, ngp = self._pref(nb_pref, x, m)
        if self.mem_k:
            mx, mmk, mv, mg = self._mem(p, x, m)
            if self.nb_gap:
                nmx, nmmk, nmv, nmg = self._mem(qn, x, m)
        flipped = self.train and random.random() < 0.5
        if flipped:                                        # 쌍 전체에 동일 flip
            x, m, r = x.flip(-1), m.flip(-1), r.flip(-1)
            if self.mem_k:
                mx, mmk = mx.flip(-1), mmk.flip(-1)
                if self.nb_gap:
                    nmx, nmmk = nmx.flip(-1), nmmk.flip(-1)
            if self.nb_gap:
                xn, mn = xn.flip(-1), mn.flip(-1)
            if self.pref_gap:
                pr, prm = pr.flip(-1), prm.flip(-1)
                if self.nb_gap:
                    npr, nprm = npr.flip(-1), nprm.flip(-1)
        out.update({'x': x, 'mask': m, 'ref': r, 'has_ref': has})
        if self.anc_dir:                                   # anchor 위치 의사 GT: SAM2 신경 track 마스크 무게중심 (flip 반영)
            c = _anc_target(p, self.anc_dir)
            out['anc'] = torch.tensor([0.5, 0.5] if c is None else [1 - c[0] if flipped else c[0], c[1]])
            out['has_anc'] = torch.tensor(float(c is not None))
        if self.nb_gap:
            out.update({'nb': xn, 'nb_mask': mn, 'has_nb': hn})
        if self.pref_gap:
            out.update({'pref': pr, 'pref_mask': prm, 'has_pref': hp, 'pref_gap': gp})
            if self.ref_flow:
                out['pref_flow'] = ref_flow(x, pr) if hp > 0 else torch.zeros(2, *x.shape[-2:])
            if self.nb_gap:
                out.update({'nb_pref': npr, 'nb_pref_mask': nprm, 'nb_has_pref': nhp, 'nb_pref_gap': ngp})
                if self.ref_flow:
                    out['nb_pref_flow'] = ref_flow(xn, npr) if nhp > 0 else torch.zeros(2, *x.shape[-2:])
        if self.mem_k:
            out.update({'mem': mx, 'mem_mask': mmk, 'mem_v': mv, 'mem_gap': mg})
            if self.nb_gap:
                out.update({'nb_mem': nmx, 'nb_mem_mask': nmmk, 'nb_mem_v': nmv, 'nb_mem_gap': nmg})
        return out


VisibleDS = SourceDS                                        # 하위 호환 (vis -> ani)


class TargetDS(Dataset):
    """판별기용 실제 target 클래스 프레임."""
    def __init__(self, split, train=False, cls='anisotropic'):
        from common import list_frames
        self.paths = list_frames(split, SEM2NAME[cls])
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        x = to_tensor(load_input(self.paths[k])[0])
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
        return x


AnisoDS = TargetDS


class ClipDS(Dataset):
    """SAM2 식 클립 학습용: 같은 비디오의 source/target 클래스 프레임(시간순)에서 연속 T 프레임 (인접 간격 <= max_step).
    프레임마다 x, mask(prior), ref(source 프레임의 pseudo-pair: 가장 가까운 target 프레임 |Δ|<=max_gap), has_ref, is_src.
    train: stride T//2 + 클립 전체 동일 flip, eval 은 사용하지 않음(스트리밍 평가는 videos() 로 직접)."""
    def __init__(self, split, T=4, train=True, max_gap=5, max_step=3, mask_dir=None, src='anisotropic', ref='visible'):
        self.T, self.train, self.mask_dir, self.split = T, train, mask_dir, split
        sn, rn = SEM2NAME[src], SEM2NAME[ref]
        self.clips = []
        for vid, fr in videos(split, [sn, rn]).items():
            r_idx = np.array([i for i, _, c in fr if c == rn]); r_path = [p for _, p, c in fr if c == rn]
            items = []
            for i, p, c in fr:
                rp = None
                if c == sn and len(r_idx):
                    j = int(np.argmin(np.abs(r_idx - i)))
                    if abs(int(r_idx[j]) - i) <= max_gap:
                        rp = r_path[j]
                items.append((i, p, rp, int(c == sn)))
            for s in range(0, len(items) - T + 1, max(1, T // 2) if train else T):
                w = items[s:s + T]
                if all(w[k + 1][0] - w[k][0] <= max_step for k in range(T - 1)):
                    self.clips.append(w)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, k):
        xs, ms, rs, hs, ss = [], [], [], [], []
        for _, p, rp, s in self.clips[k]:
            im, _ = load_input(p); g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            xs.append(to_tensor(im)); ms.append(torch.from_numpy(_prior_mask(p, g, self.mask_dir, self.split))[None].float())
            rs.append(to_tensor(load_input(rp)[0]) if rp else torch.zeros_like(xs[-1])); hs.append(float(rp is not None)); ss.append(float(s))
        x, m, r = torch.stack(xs), torch.stack(ms), torch.stack(rs)
        if self.train and random.random() < 0.5:
            x, m, r = x.flip(-1), m.flip(-1), r.flip(-1)
        return {'x': x, 'mask': m, 'ref': r, 'has_ref': torch.tensor(hs), 'is_src': torch.tensor(ss)}
