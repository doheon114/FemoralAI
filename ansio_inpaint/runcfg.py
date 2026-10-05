"""방향(vis2ani / ani2vis) 별 설정 해석 + 체크포인트 args 로부터 Generator 생성.
기존 체크포인트(direction 등 인자 없음)는 vis2ani 기본값으로 해석되어 호환된다."""
from models import LightUNetGenerator

# direction -> (source 의미 클래스, target 의미 클래스)
DIRECTIONS = {'vis2ani': ('visible', 'anisotropic'), 'ani2vis': ('anisotropic', 'visible')}

# 방향별 기본값.
#  target_p / src_p : 성공 기준 P(target) >= target_p and P(source) <= src_p
#    ani2vis: 분류기는 실제 Visible 프레임에도 P(vis) 를 최대 ~0.8 로만 주므로(중앙값 ~0.7) 0.85 는 도달 불가
#             -> 실제 Visible 수준(P(vis)>=0.60, P(ani)<=0.20) 을 기준으로 사용.
#  gain / offset / alpha : physical 파라미터 범위 (alpha<0 = unsharp 선명화, >0 = blur)
DEFAULTS = {
    'vis2ani': dict(target_p=0.85, src_p=0.10, gain_range=(0.25, 1.25), max_offset=0.3, alpha_range=(0.0, 1.0)),
    'ani2vis': dict(target_p=0.60, src_p=0.20, gain_range=(0.8, 1.6), max_offset=0.1, alpha_range=(-1.0, 0.5)),
}


def resolve(t):
    """argparse Namespace / SimpleNamespace -> 누락 항목을 방향별 기본값으로 채운 dict."""
    d = dict(vars(t))
    d.setdefault('direction', 'vis2ani')
    dft = DEFAULTS[d['direction']]
    if d['direction'] == 'vis2ani':                          # 구버전 인자 이름 호환
        d.setdefault('target_p', d.get('target_ani', dft['target_p']))
        d.setdefault('src_p', d.get('target_vis', dft['src_p']))
    for k, v in dft.items():
        if d.get(k) is None:
            d[k] = v
    d['src_sem'], d['tgt_sem'] = DIRECTIONS[d['direction']]
    d.setdefault('param', 'free')
    d.setdefault('anchor_r', 0.0)
    d['sca_at'] = tuple(d.get('sca_at') or ())             # 구버전(SCA 없음) 호환
    d.setdefault('sca_iters', 3)
    d.setdefault('sca_prior', 'anchor')
    d.setdefault('up_after', False)
    return d


def build_generator(d, pretrained=False):
    G = _build_generator(d, pretrained)
    G.anc_tau = d.get('anc_tau', 1.0) or 1.0                # anchor soft-argmax 온도
    G.anc_local = d.get('anc_local', 0) or 0                # anchor 국소 soft-argmax 반경 (셀)
    return G


def _build_generator(d, pretrained=False):
    return LightUNetGenerator(max_delta=d.get('max_delta', 1.0), pretrained=pretrained, param=d['param'],
                              anchor_r=d['anchor_r'], gain_range=tuple(d['gain_range']),
                              max_offset=d['max_offset'], alpha_range=tuple(d['alpha_range']),
                              sca_at=tuple(d.get('sca_at', ())), sca_iters=d.get('sca_iters', 3),
                              sca_prior=d.get('sca_prior', 'anchor'), up_after=bool(d.get('up_after', False)),
                              ref_in=bool(d.get('ref_in', 0)), ref_gap=d.get('ref_gap', 30),
                              mem_k=d.get('mem_k', 0) or 0, mem_gap=d.get('mem_gap', 30),
                              detail_max=d.get('detail_max', 0.25),
                              detail_gate=bool(d.get('detail_gate', 0)),
                              phys_stride=d.get('phys_stride', 8) or 8,
                              phys_blur=d.get('phys_blur', 0.0) or 0.0,
                              mem_mode=d.get('mem_mode', 'slot') or 'slot', mem_at=tuple(d.get('mem_at') or (8, 16)),
                              mem_dist=bool(d.get('mem_dist', 1)),
                              blur_sigma=d.get('blur_sigma', 3.0) or 3.0,
                              encoder=d.get('encoder', 'small') or 'small',
                              mask_head=bool(d.get('mask_head', 0)), s2m=d.get('s2m', 0) or 0,
                              tex=d.get('tex', 0) or 0, tex_flow=d.get('tex_flow', 0.15) or 0.15,
                              enhance_only=d.get('enhance_only', 0) or 0)


def src_tgt_probs(C, logits, d):
    """-> (p_src, p_tgt, p_vis, p_ani, p_abs)"""
    pv, pa, pb = C.sem_probs(logits)
    sem = {'visible': pv, 'anisotropic': pa}
    return sem[d['src_sem']], sem[d['tgt_sem']], pv, pa, pb
