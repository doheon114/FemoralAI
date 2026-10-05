"""Lightweight UNet Generator (MobileNetV3-Small encoder) + Localized Residual composition.

입력 : I_vis [B,3,H,W] in [-1,1]  +  prior mask M [B,1,H,W] in [0,1]
출력 : ΔI [B,3,H,W] (tanh * max_delta)  +  학습형 위치 마스크 m̂ [B,1,H,W] (sigmoid)
합성 : M_eff = M ⊙ m̂ ,   I_out = clamp(I_vis + ΔI ⊙ M_eff, -1, 1)

m̂ 는 Grad-CAM 없이 분류기 점수 변화(L_cls)와 면적/TV 정규화만으로 "어디를 바꿔야 anisotropic 으로
판정되는지"를 스스로 학습한다. H, W 는 32 의 배수여야 한다(224 OK).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DSConv(nn.Sequential):
    """depthwise 3x3 + pointwise 1x1 (BN, ReLU)"""
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(DSConv(cin + cskip, cout), DSConv(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


def _gauss1d(sigma):
    r = int(3 * sigma + 0.5)
    g = torch.exp(-(torch.arange(-r, r + 1).float() ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class SpatialCompletionAdapter(nn.Module):
    """Amodal-SAM 식 Spatial Completion Adapter (encoder 중간 삽입, feature space completion).
    입력: encoder feature E [B,C,h,w] + downsample 된 target prior M_spec [B,1,h,w].
    iters 회 반복:  G = σ(F_gate([E, M])),  O = G ⊙ φ(F_feat([E, M])),  E <- E + O
    F_* = 1x1 -> dilated depthwise 3x3 -> 1x1 (넓은 수용야로 가려진 위치를 주변 feature 로 보완).
    F_feat 마지막 conv 는 0 초기화 -> 시작 시 identity (기존 체크포인트에 그대로 삽입 가능)."""
    def __init__(self, c, iters=3, dilation=2):
        super().__init__()
        h = max(16, c)

        def branch(zero):
            last = nn.Conv2d(h, c, 1)
            if zero:
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            return nn.Sequential(nn.Conv2d(c + 1, h, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(h, h, 3, padding=dilation, dilation=dilation, groups=h), nn.ReLU(inplace=True),
                                 last)
        self.gate = nn.ModuleList(branch(False) for _ in range(iters))
        self.feat = nn.ModuleList(branch(True) for _ in range(iters))
        self.last_gate = None

    def forward(self, e, m):
        m = F.interpolate(m, size=e.shape[-2:], mode='bilinear', align_corners=False) if m.shape[-2:] != e.shape[-2:] else m
        for fg, ff in zip(self.gate, self.feat):
            z = torch.cat([e, m.to(e.dtype)], 1)
            g = torch.sigmoid(fg(z))
            e = e + g * F.gelu(ff(z))
        self.last_gate = g
        return e


class RefFuse(nn.Module):
    """과거 참조 프레임(가장 최근 target 클래스 프레임) feature 주입.
    E <- E + v · Conv1x1([E_ref, E - E_ref, gap/K])   (0 초기화 -> 시작 시 identity, 기존 체크포인트 그대로 로드)
    v: 참조 유효(0/1), gap/K: 참조까지의 프레임 거리(멀수록 덜 믿도록 학습)."""
    def __init__(self, c, dilation=2):
        super().__init__()
        h = max(16, c)
        last = nn.Conv2d(h, c, 1)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        # 1x1 -> dilated depthwise 3x3 (참조 프레임과의 위치 어긋남 흡수) -> 1x1(0 초기화)
        self.conv = nn.Sequential(nn.Conv2d(2 * c + 1, h, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(h, h, 3, padding=dilation, dilation=dilation, groups=h), nn.ReLU(inplace=True),
                                  last)

    def forward(self, e, e_ref, R):
        gn = R['gn'].to(e.dtype).expand(-1, 1, e.shape[-2], e.shape[-1])
        return e + R['v'].to(e.dtype) * self.conv(torch.cat([e_ref.to(e.dtype), e - e_ref.to(e.dtype), gn], 1))


class MemFuse(nn.Module):
    """과거 프레임 feature 메모리 읽기 (인과적: 현재보다 앞선 K 프레임, 분류기 P(vis) 게이트).
    위치별 K-slot attention:  a_k = softmax_k( <Wq E, Wk E_k>/√d + w_g·gap_k )  (무효 slot 마스킹)
    R = Σ a_k E_k ,  E <- E + v · Conv([R, E - R, gap_read])   (마지막 conv 0 초기화 -> 시작 시 identity)"""
    def __init__(self, c, dilation=2):
        super().__init__()
        self.d = max(16, c // 2)
        self.q = nn.Conv2d(c, self.d, 1)
        self.k = nn.Conv2d(c, self.d, 1)
        self.w_gap = nn.Parameter(torch.tensor(-1.0))      # 오래된 메모리일수록 낮은 가중(학습)
        self.fuse = RefFuse(c, dilation)
        self.last_att = None

    def forward(self, e, mem, M):
        B, K = mem.shape[:2]
        q = self.q(e).float()                                              # [B,d,h,w]
        k = self.k(mem.flatten(0, 1).to(e.dtype)).float().view(B, K, self.d, *e.shape[-2:])
        sc = (q.unsqueeze(1) * k).sum(2) / self.d ** 0.5 + self.w_gap * M['gn'].view(B, K, 1, 1)
        valid = M['v'].view(B, K, 1, 1) > 0.5
        sc = sc.masked_fill(~valid, -1e4)
        a = torch.softmax(sc, 1) * valid                                   # [B,K,h,w] (전부 무효면 0)
        self.last_att = a
        r = (a.unsqueeze(2) * mem.float()).sum(1)
        gn = (a * M['gn'].view(B, K, 1, 1)).sum(1, keepdim=True)
        v = (M['v'].sum(1) > 0).float().view(B, 1, 1, 1)
        return self.fuse(e, r.to(e.dtype), {'v': v, 'gn': gn})


class MemXAttn(nn.Module):
    """과거 VIS 메모리 공간 cross-attention (MemFuse 재설계).
    MemFuse 는 같은 픽셀 위치의 K slot 끼리만 attention -> 프로브/신경이 움직이면 정렬 불가.
    여기선 현재 feature 의 각 위치(query)가 K 프레임의 *모든 위치*(key) 를 본다:
      logit = <Wq E(p), Wk E_k(p')>/√d  − softplus(w_d)·|p − p'|²  + w_g·gap_k   (무효 slot 마스킹, multi-head)
    R = Σ a·Wv E_k  -> RefFuse (0 초기화 -> 시작 시 identity).  last_read = Σ ā·E_k (원 feature, 메모리 일관성 loss target)."""
    def __init__(self, c, heads=4, dh=16, dilation=2, dist=True):
        super().__init__()
        self.h, self.dh, self.dist = heads, dh, dist         # dist=False: 거리 prior 없음 (픽셀 위치 무관, feature 유사도로만 대응)
        self.q = nn.Conv2d(c, heads * dh, 1)
        self.k = nn.Conv2d(c, heads * dh, 1)
        self.v = nn.Conv2d(c, heads * dh, 1)
        self.o = nn.Conv2d(heads * dh, c, 1)
        self.w_dist = nn.Parameter(torch.full((heads,), 17.8))  # 거리 벌점(head 별): softplus·10≈178 = 1/(2σ²), σ≈0.053(/8 에서 ~1.5셀). 학습으로 넓힐 수 있음
        self.w_gap = nn.Parameter(torch.tensor(-1.0))
        self.fuse = RefFuse(c, dilation)
        self.last_read = None
        self.last_att = None

    def forward(self, e, mem, M):
        B, K, C, h, w = mem.shape
        N, H, D = h * w, self.h, self.dh
        q = self.q(e).float().view(B, H, D, N).transpose(2, 3)                            # [B,H,N,D]
        mf = mem.flatten(0, 1).to(e.dtype)
        k = self.k(mf).float().view(B, K, H, D, N).permute(0, 2, 1, 4, 3).reshape(B, H, K * N, D)
        v = self.v(mf).float().view(B, K, H, D, N).permute(0, 2, 1, 4, 3).reshape(B, H, K * N, D)
        ys, xs = torch.meshgrid((torch.arange(h, device=e.device) + 0.5) / h, (torch.arange(w, device=e.device) + 0.5) / w, indexing='ij')
        pos = torch.stack([ys.flatten(), xs.flatten()], 1)                                # [N,2]
        d2 = torch.cdist(pos, pos) ** 2                                                    # [N,N]
        bias = (-F.softplus(self.w_dist).view(1, H, 1, 1, 1) * d2.view(1, 1, N, 1, N) * 10.0) if self.dist \
            else torch.zeros(1, 1, N, 1, N, device=e.device)
        bias = bias + (self.w_gap * M['gn']).view(B, 1, 1, K, 1)
        valid = (M['v'] > 0.5).view(B, 1, 1, K, 1)
        bias = bias.masked_fill(~valid, -1e4).expand(B, H, N, K, N).reshape(B, H, N, K * N)                    # 전부 무효면 균일(fuse v=0 으로 꺼짐)
        att = torch.softmax(q @ k.transpose(-1, -2) / D ** 0.5 + bias, -1)                # [B,H,N,K*N]
        self.last_att = att.detach().view(B, H, N, K, N).sum(-1).mean(1)                   # [B,N,K] slot 별 사용량
        r = self.o((att @ v).transpose(2, 3).reshape(B, H * D, h, w)).float()
        # 일관성 loss target: head 평균 attention 으로 읽은 *원 메모리 feature* (투영 전, 같은 feature 공간)
        raw = mem.float().view(B, K, C, N).permute(0, 2, 1, 3).reshape(B, C, K * N)
        self.last_read = (raw @ att.mean(1).transpose(1, 2)).view(B, C, h, w)
        vv = (M['v'].sum(1) > 0).float().view(B, 1, 1, 1)
        gn = ((M['gn'] * M['v']).sum(1) / M['v'].sum(1).clamp(min=1)).view(B, 1, 1, 1)
        return self.fuse(e, r.to(e.dtype), {'v': vv, 'gn': gn})


def _sincos2d(h, w, d, device):
    """2D sin/cos 위치 인코딩 [h*w, d] (d 는 4 의 배수)."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    q = d // 4
    f = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    ay, ax = ys.flatten()[:, None] * f, xs.flatten()[:, None] * f
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1)


class MemAttnSAM(nn.Module):
    """SAM2 식 memory attention (과거 프레임 메모리로 현재 feature 조정).
    memory encoder : M_k = E_k + MaskEmb(m̂_k)    (m̂_k = 메모리 프레임에 대한 G 자신의 편집 마스크 예측, no-grad)
                     key 에 2D 위치 인코딩 + 시간 임베딩(gap_k) 을 더함 (SAM2 temporal pos enc)
    memory attention (현재 토큰 X, pre-LN transformer 1 layer):
        X <- X + SelfAttn(X)          (현재 프레임 공간 문맥)
        X <- X + CrossAttn(X, {M_k})  (K 프레임 × 모든 위치, 무효 slot 마스킹)
        X <- X + MLP(X)
    E <- E + v · Wo(X)   (Wo 0 초기화 -> 시작 시 identity, 유효 메모리 없는 샘플(v=0)은 원 feature 그대로)."""
    def __init__(self, c, gap_max=30, d=64, heads=4):
        super().__init__()
        self.d, self.h, self.gap_max = d, heads, gap_max
        self.inp = nn.Conv2d(c, d, 1)
        self.mem_in = nn.Conv2d(c, d, 1)
        self.mask_emb = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, d, 1))
        self.t_emb = nn.Embedding(gap_max + 1, d)
        self.ln1, self.ln2, self.ln3, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca_q, self.ca_k, self.ca_v, self.ca_o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.out = nn.Conv2d(d, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.last_att = None

    def forward(self, e, mem, M):
        B, K, C, h, w = mem.shape
        N, H, d = h * w, self.h, self.d
        pos = _sincos2d(h, w, d, e.device).to(e.dtype)                                  # [N,d]
        x = self.inp(e).flatten(2).transpose(1, 2)                                       # [B,N,d]
        mk = F.interpolate(M['mask'].flatten(0, 1).to(e.dtype), size=(h, w), mode='area')    # [BK,1,h,w]
        mt = (self.mem_in(mem.flatten(0, 1).to(e.dtype)) + self.mask_emb(mk)).flatten(2).transpose(1, 2)   # [BK,N,d]
        mt = self.ln_m(mt).view(B, K, N, d)
        gi = M['gap'].round().long().clamp(0, self.gap_max)                             # [B,K]
        mkey = (mt + pos + self.t_emb(gi)[:, :, None]).reshape(B, K * N, d)
        mval = mt.reshape(B, K * N, d)
        # self-attn
        y = self.ln1(x) + pos
        x = x + self.sa(y, y, self.ln1(x), need_weights=False)[0]
        # cross-attn to memory
        valid = M['v'] > 0.5                                                             # [B,K]
        vv = valid.any(1)
        am = valid[:, :, None].expand(B, K, N).reshape(B, 1, 1, K * N)
        am = am | ~vv.view(B, 1, 1, 1)                                                   # 전부 무효면 전체 허용(출력은 v=0 으로 꺼짐)
        split = lambda t: t.view(B, -1, H, d // H).transpose(1, 2)
        q = split(self.ca_q(self.ln2(x) + pos))
        r = F.scaled_dot_product_attention(q, split(self.ca_k(mkey)), split(self.ca_v(mval)), attn_mask=am)
        x = x + self.ca_o(r.transpose(1, 2).reshape(B, N, d))
        x = x + self.mlp(self.ln3(x))
        with torch.no_grad():                                                            # 진단용 slot 사용량 [B,K]
            a = torch.softmax((q @ split(self.ca_k(mkey)).transpose(-1, -2) / (d // H) ** 0.5).float().masked_fill(~am, -1e4), -1)
            self.last_att = a.mean((1, 2)).view(B, K, N).sum(-1)
        o = self.out(x.transpose(1, 2).reshape(B, d, h, w))
        return e + vv.to(e.dtype).view(B, 1, 1, 1) * o


class LightUNetGenerator(nn.Module):
    """param='physical' (기본): /8 해상도에서 gain / offset / blur-mix / mask 4개 맵만 예측 후 bilinear 업샘플.
         out01 = gain * (g + α (blur(g) - g)) + b ,  ΔI = 2 (out01 - g) (3채널 복제)   (α<0 이면 unsharp 선명화)
         -> speckle 질감은 보존하고 국소 저에코화 + 신경다발 무늬 소실(실제 anisotropy 양상)만 표현 가능,
            분류기를 속이는 고주파 적대 패턴이 구조적으로 불가능.
       param='free': 전해상도 픽셀별 ΔI (tanh * max_delta).
       param='hybrid': physical(/8, 저주파 밝기·대비) + 전해상도 detail residual (tanh * detail_max, 1ch)
         -> /8 파라미터만으로는 못 그리는 신경다발 내부 고주파 구조를 마스크 안에서 추가.
    anchor_r > 0: 단일 위치 anchor window.  c = /8 heatmap soft-argmax (한 점),
         m̂ = sigmoid(map) ⊙ exp(-|p - c|² / 2R²)  -> 모양은 자유, 위치는 한 곳(멀리 떨어진 덩어리 억제).
    sca_at (예: (4, 8, 16)): 해당 stride 의 encoder tap 뒤에 SpatialCompletionAdapter 삽입 (shallow/middle/deep).
         2-pass: pass1(SCA off) 로 target prior M_spec (= M ⊙ anchor window, 또는 M ⊙ m̂) 을 구하고,
         pass2 에서 SCA 가 M_spec 을 받아 가려진(anisotropic) 신경 feature 를 feature space 에서 completion."""
    def __init__(self, in_ch=3, out_ch=3, max_delta=1.0, pretrained=True, dec_ch=(96, 64, 32, 24, 16),
                 mask_bias=-2.0, param='physical', blur_sigma=3.0, gain_range=(0.25, 1.25), max_offset=0.3,
                 anchor_r=0.0, alpha_range=(0.0, 1.0), sca_at=(), sca_iters=3, sca_prior='anchor', up_after=False,
                 ref_in=False, ref_gap=30, mem_k=0, mem_gap=30, detail_max=0.25, detail_gate=False, phys_stride=8, phys_blur=0.0,
                 mem_mode='slot', mem_at=(8, 16), mem_dist=True, encoder='small', mask_head=False, s2m=0, tex=0, tex_flow=0.15, enhance_only=0, **_):
        super().__init__()
        self.tex, self.tex_flow = tex, tex_flow
        self.enhance_only = enhance_only                    # 1: ΔI >= 0 (밝히기만, 지우기/어둡게 금지)             # tex: 과거 실제 VIS 참조 프레임을 warp 해 질감(HF)·저주파(LF) 잔차를 물리 맵 위에 더함
        self.detail_max, self.detail_gate = detail_max, detail_gate
        self.phys_stride = phys_stride
        self.phys_blur = phys_blur                          # >0: 물리 맵(σ 단위=셀) 가우시안 평활 -> 셀별 독립 진동(계단/벌집) 차단
        if phys_blur > 0:
            self.register_buffer('pblur_k', _gauss1d(phys_blur), persistent=False)
        self.ref_in, self.ref_gap = ref_in, ref_gap
        self.mem_k, self.mem_gap, self.mem_mode = mem_k, mem_gap, mem_mode
        self.sca_prior, self.up_after = sca_prior, up_after
        self.max_delta, self.param, self.anchor_r = max_delta, param, anchor_r
        self.gain_lo, self.gain_span, self.max_offset = gain_range[0], gain_range[1] - gain_range[0], max_offset
        self.alpha_lo, self.alpha_span = alpha_range[0], alpha_range[1] - alpha_range[0]
        self.register_buffer('blur_k', _gauss1d(blur_sigma), persistent=False)
        self.encoder = encoder                              # 'small' | 'large' (MobileNetV3)
        ctor, W = {'small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
                   'large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights)}[encoder]
        try:
            enc = ctor(weights=W.IMAGENET1K_V1 if pretrained else None).features
        except Exception as e:                              # 오프라인 등
            print(f'[generator] pretrained load failed ({e}); using random init')
            enc = ctor(weights=None).features

        # 첫 conv 를 (in_ch + 1 mask) 채널로 확장: RGB 가중치 복사, mask 채널은 평균 가중치로 초기화
        c0 = enc[0][0]
        new = nn.Conv2d(in_ch + 1, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = c0.weight
            new.weight[:, 3:] = c0.weight.mean(1, keepdim=True)
        enc[0][0] = new
        self.enc = enc

        # stride 별 마지막 블록을 skip tap 으로 자동 탐지 (/2,/4,/8,/16,/32)
        with torch.no_grad():
            x = torch.zeros(1, in_ch + 1, 224, 224)
            taps, chs, prev = {}, {}, None
            for i, blk in enumerate(self.enc):
                x = blk(x)
                taps[x.shape[-1]] = i
                chs[x.shape[-1]] = x.shape[1]
        sizes = sorted(taps, reverse=True)                  # [112,56,28,14,7]
        self.tap_idx = [taps[s] for s in sizes]
        enc_ch = [chs[s] for s in sizes]                    # e.g. [16,16,24,48,576]
        stride_ch = {224 // s: (taps[s], chs[s]) for s in sizes}   # {2: (idx, ch), 4: ..., 32: ...}
        self.sca_idx = [stride_ch[st][0] for st in sca_at]
        self.sca = nn.ModuleDict({str(stride_ch[st][0]): SpatialCompletionAdapter(stride_ch[st][1], sca_iters)
                                  for st in sca_at})
        # ref_in: SCA 위치 + /8 skip 위치에서 과거 참조 프레임 feature 주입 (pass1 위치 찾기, pass2 completion 모두)
        # mem_k: 같은 위치(SCA + /8)에서 과거 K 프레임 feature 메모리 attention 읽기 (ref_in 과 배타)
        self.ref_idx = sorted(set(self.sca_idx) | {stride_ch[8][0]}) if (ref_in or mem_k) else []
        # s2m: SAM2 메모리 구조(memory encoder/bank/attention) 를 /16 feature 에 이식 (1=SAM2 가중치, 2=무작위 초기화)
        self.s2m_idx, self.s2m = stride_ch[16][0], None
        if s2m:
            from .sam2mem import SAM2Memory
            self.s2m = SAM2Memory(stride_ch[16][1], pretrained=(s2m == 1))
        self._s2_bank, self._s2_pre = None, None
        ch_of = lambda i: [c for (j, c) in stride_ch.values() if j == i][0]
        if mem_k and mem_mode == 'sam':                   # SAM2 식 memory attention: /8·/16 (mem_at)
            self.ref_idx = sorted(stride_ch[s][0] for s in mem_at)
            self.mem_fuse = nn.ModuleDict({str(i): MemAttnSAM(ch_of(i), mem_gap) for i in self.ref_idx})
        elif mem_k and mem_mode == 'xattn':                 # 공간 cross-attention: /8·/16 (/4 는 K·56⁴ 로 과대)
            self.ref_idx = sorted(stride_ch[s][0] for s in mem_at)
            self.mem_fuse = nn.ModuleDict({str(i): MemXAttn(ch_of(i), dist=mem_dist) for i in self.ref_idx})
        elif mem_k:
            self.mem_fuse = nn.ModuleDict({str(i): MemFuse(ch_of(i)) for i in self.ref_idx})
        else:
            self.ref_fuse = nn.ModuleDict({str(i): RefFuse(ch_of(i)) for i in self.ref_idx})

        d = dec_ch
        self.up4 = UpBlock(enc_ch[4], enc_ch[3], d[0])      # /32 -> /16
        self.up3 = UpBlock(d[0], enc_ch[2], d[1])           # -> /8
        if anchor_r > 0:
            self.anchor_heat = nn.Conv2d(d[1], 1, 3, padding=1)  # 0 init -> 균일 heatmap -> 중심에서 시작
            nn.init.zeros_(self.anchor_heat.weight); nn.init.zeros_(self.anchor_heat.bias)
        self.use_mask_head = bool(mask_head) and anchor_r > 0
        if self.use_mask_head:                              # Gaussian 창 대신: [F28, anchor heatmap] -> 28² soft mask 직접 예측
            last = nn.Conv2d(32, 1, 3, padding=1)
            nn.init.zeros_(last.weight); nn.init.constant_(last.bias, mask_bias)
            self.mask_head = nn.Sequential(DSConv(d[1] + 1, 32), last)
        if param in ('physical', 'hybrid'):
            if phys_stride in (4, 2, 1):                    # up2 추가: 물리 맵을 /4 (56²) 에서 예측 (s4 skip 사용)
                self.up2_phys = UpBlock(d[1], enc_ch[1], d[2])
            if phys_stride in (2, 1):                       # up1 추가: 물리 맵을 /2 에서 예측 (s2 skip 사용)
                self.up1_phys = UpBlock(d[2], enc_ch[0], d[3])
            if phys_stride == 1:                            # up0 추가: 전해상도 U-Net (입력 자체를 skip) -> 물리 맵 @ /1
                self.up0_phys = UpBlock(d[3], in_ch + 1, d[4])
            self.head_phys = nn.Conv2d({8: d[1], 4: d[2], 2: d[3], 1: d[4]}[phys_stride], 4, 3, padding=1)   # gain, offset, blur-mix, mask  (@ /8 or /4)
            nn.init.zeros_(self.head_phys.weight)
            with torch.no_grad():                           # 시작 시 identity: gain=1, b=0, α≈0
                g0 = (1.0 - self.gain_lo) / self.gain_span
                a0 = -4.0 if self.alpha_lo >= 0 else float(torch.logit(torch.tensor(-self.alpha_lo / self.alpha_span)))
                self.head_phys.bias.copy_(torch.tensor([float(torch.logit(torch.tensor(g0))), 0.0, a0, mask_bias]))
        if tex:                                             # /8: flow(2) + β_hf + β_lf  (시작 시 flow=0, β≈0 -> v4 와 동일)
            self.head_tex = nn.Conv2d(d[1], 4, 3, padding=1)
            nn.init.zeros_(self.head_tex.weight)
            with torch.no_grad():
                self.head_tex.bias.copy_(torch.tensor([0.0, 0.0, -3.0, -3.0]))
        if param == 'hybrid':                               # 전해상도 detail branch (시작 시 0 -> physical 과 동일)
            self.up2 = UpBlock(d[1], enc_ch[1], d[2])
            self.up1 = UpBlock(d[2], enc_ch[0], d[3])
            self.up0 = UpBlock(d[3], in_ch + 1, d[4])
            self.head_detail = nn.Conv2d(d[4], 1, 3, padding=1)
            nn.init.zeros_(self.head_detail.weight); nn.init.zeros_(self.head_detail.bias)
        if param == 'free':
            self.up2 = UpBlock(d[1], enc_ch[1], d[2])       # -> /4
            self.up1 = UpBlock(d[2], enc_ch[0], d[3])       # -> /2
            self.up0 = UpBlock(d[3], in_ch + 1, d[4])       # -> /1 (입력 자체를 skip)
            self.head_delta = nn.Conv2d(d[4], out_ch, 3, padding=1)
            self.head_mask = nn.Conv2d(d[4], 1, 3, padding=1)
            nn.init.zeros_(self.head_delta.weight); nn.init.zeros_(self.head_delta.bias)   # 시작 시 ΔI=0 (identity)
            nn.init.zeros_(self.head_mask.weight); nn.init.constant_(self.head_mask.bias, mask_bias)

    def _blur(self, g, k=None):
        k = (self.blur_k if k is None else k).to(g.dtype)
        r, C = k.numel() // 2, g.shape[1]
        g = F.conv2d(F.pad(g, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
        return F.conv2d(F.pad(g, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)

    def _anchor_window(self, y, H, W, c=None):
        """c [B,2] (x, y in [0,1]) 가 주어지면 soft-argmax 대신 그 위치를 anchor 로 강제 (anchor search 후보 평가용)."""
        B, _, h, w = y.shape
        self._anc_logit = self.anchor_heat(y).flatten(1)                 # anchor 위치 학습(--lam_anc / --lam_asearch)용
        lgt = self._anc_logit / getattr(self, 'anc_tau', 1.0)            # tau<1: soft-argmax 의 중앙 수축 완화
        ref = getattr(self, 'anc_ref', None)                             # 추적(추론 전용): 이전 프레임 anchor [B,2] 주변 Gaussian prior
        if ref is not None and ref.shape[0] == B:                         #   log-prior 를 heatmap logit 에 더함 (anc_ref_valid=0 샘플은 prior 없음)
            gy = ((torch.arange(h, device=y.device, dtype=lgt.dtype) + 0.5) / h).view(1, h, 1)
            gx = ((torch.arange(w, device=y.device, dtype=lgt.dtype) + 0.5) / w).view(1, 1, w)
            rr = ref.to(lgt)
            lp = -((gy - rr[:, 1].view(-1, 1, 1)) ** 2 + (gx - rr[:, 0].view(-1, 1, 1)) ** 2) / (2 * self.anc_ref_sigma ** 2)
            lgt = lgt + lp.flatten(1) * self.anc_ref_valid.to(lgt).view(-1, 1)
        k = getattr(self, 'anc_local', 0)                                 # >0: argmax 셀 주변 (2k+1)² 안에서만 soft-argmax
        if k > 0:                                                         #   평평한 heatmap 의 전역 평균(=중앙) 수축 방지
            i0 = lgt.detach().argmax(1)
            iy, ix = (i0 // w).view(B, 1, 1), (i0 % w).view(B, 1, 1)
            ay = torch.arange(h, device=y.device).view(1, h, 1); ax = torch.arange(w, device=y.device).view(1, 1, w)
            near = ((ay - iy).abs() <= k) & ((ax - ix).abs() <= k)
            lgt = lgt.masked_fill(~near.flatten(1), float('-inf'))
        att = torch.softmax(lgt, 1).view(B, h, w)
        cy = (att.sum(2) * ((torch.arange(h, device=y.device, dtype=y.dtype) + 0.5) / h)).sum(1)
        cx = (att.sum(1) * ((torch.arange(w, device=y.device, dtype=y.dtype) + 0.5) / w)).sum(1)
        if c is not None:
            cx, cy = c[:, 0].to(y.dtype), c[:, 1].to(y.dtype)
        Y = ((torch.arange(H, device=y.device, dtype=y.dtype) + 0.5) / H).view(1, H, 1)
        X = ((torch.arange(W, device=y.device, dtype=y.dtype) + 0.5) / W).view(1, 1, W)
        d2 = (Y - cy.view(B, 1, 1)) ** 2 + (X - cx.view(B, 1, 1)) ** 2
        return torch.exp(-d2 / (2 * self.anchor_r ** 2)).unsqueeze(1), torch.stack([cx, cy], 1)

    def _tex(self, y, g, ref, ref_v, ref_flow=None):
        """/8 맵 -> flow (이미지 폭 비율, ±tex_flow) 로 참조 gray 를 warp,
        tex = β_hf (w - blur w) + β_lf (blur w - blur g).  ref_v=0 샘플은 0."""
        t = self.head_tex(y)
        q = torch.cat([torch.tanh(t[:, :2]) * self.tex_flow, torch.sigmoid(t[:, 2:4])], 1)
        q = F.interpolate(q, size=g.shape[-2:], mode='bilinear', align_corners=False)
        if ref_flow is not None:                            # 고정 정합 flow (DIS) 사용: 학습 flow 채널 무시 -> 구조를 끌어오는 악용 차단
            q = torch.cat([ref_flow.to(q), q[:, 2:]], 1)
        B, _, H, W = g.shape
        gy = (torch.arange(H, device=g.device, dtype=g.dtype) + 0.5) / H * 2 - 1
        gx = (torch.arange(W, device=g.device, dtype=g.dtype) + 0.5) / W * 2 - 1
        base = torch.stack(torch.meshgrid(gx, gy, indexing='xy'), -1).unsqueeze(0)   # [1,H,W,2] (x, y)
        grid = base + 2 * q[:, :2].permute(0, 2, 3, 1).to(g.dtype)
        gr = (ref[:, :1] + 1) * 0.5
        w = F.grid_sample(gr.to(g.dtype), grid, mode='bilinear', padding_mode='border', align_corners=False)
        bw = self._blur(w)
        tex = q[:, 2:3] * (w - bw) + q[:, 3:4] * (bw - self._blur(g))
        self._tex_stat = (q[:, :2], q[:, 2:3], q[:, 3:4])
        v = ref_v.to(g.dtype).view(B, 1, 1, 1) if ref_v is not None else 1.0
        return tex * v

    def _mask_from_head(self, y, H, W):
        """28² feature + (max 정규화) anchor heatmap -> sigmoid mask @28² -> bilinear 업샘플 @H×W. Gaussian 창 없음."""
        B, _, h, w = y.shape
        att = torch.softmax(self.anchor_heat(y).flatten(1), 1).view(B, 1, h, w)
        att = att / (att.amax((2, 3), keepdim=True) + 1e-6)
        m28 = torch.sigmoid(self.mask_head(torch.cat([y, att], 1)))
        return F.interpolate(m28, size=(H, W), mode='bilinear', align_corners=False)

    def _encode(self, inp, m_spec=None, keep=None, R=None, collect=None):
        """m_spec 이 주어지면 SCA 적용. keep: SCA 위치 feature 를 모을 list (feature completion loss 용).
        R: 참조 프레임 정보 dict(feats={idx: E_ref}, v, gn) -> ref_idx 위치에서 RefFuse. collect: ref_idx feature 수집용 dict."""
        feats, h = [], inp
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if collect is not None and i in self.ref_idx:
                collect[i] = h
            if R is not None and i in self.ref_idx:
                h = (self.mem_fuse[str(i)](h, R['feats'][i], R) if self.mem_k
                     else self.ref_fuse[str(i)](h, R['feats'][i], R))
            if self.s2m is not None and i == self.s2m_idx and collect is None:
                h, self._s2_pre = self.s2m.read(h, self._s2_bank or [])
            if m_spec is not None and i in self.sca_idx:
                h = self.sca[str(i)](h, m_spec)
            if keep is not None and i in self.sca_idx:
                keep.append(h)
            if i in self.tap_idx:
                feats.append(h)
        return feats

    def _decode8(self, feats):
        s2, s4, s8, s16, s32 = feats
        return self.up3(self.up4(s32, s16), s8)

    def _phys_feat(self, y, s4, s2=None, inp=None):
        if self.phys_stride == 8:
            return y
        y = self.up2_phys(y, s4)
        if self.phys_stride == 4:
            return y
        y = self.up1_phys(y, s2)
        return self.up0_phys(y, inp) if self.phys_stride == 1 else y

    def _mask_logit8(self, y, s4=None, s2=None, inp=None):
        return self.head_phys(self._phys_feat(y, s4, s2, inp))[:, 3:4] if self.param in ('physical', 'hybrid') else None

    @torch.no_grad()
    def sca_targets(self, x, m):
        """SCA 위치에서의 plain(SCA off) encoder feature. pseudo-pair target 클래스 프레임의 completion target."""
        keep = []
        self._encode(torch.cat([x, m], 1), None, keep)
        return keep

    def encode_ref(self, ref, ref_m, ref_v, ref_gap):
        """참조 프레임 [B,3,H,W] + 조직 마스크 -> R dict. ref_v [B] (0/1), ref_gap [B] (프레임 수)."""
        col = {}
        self._encode(torch.cat([ref, ref_m], 1), collect=col)
        B = ref.shape[0]
        v = ref_v.float().view(B, 1, 1, 1)
        gn = (ref_gap.float().clamp(0, self.ref_gap) / self.ref_gap).view(B, 1, 1, 1) * v
        return {'feats': col, 'v': v, 'gn': gn}

    @torch.no_grad()
    def encode_mem(self, mem, mem_m, mem_v, mem_gap):
        """과거 K 프레임 [B,K,3,H,W] + 마스크 [B,K,1,H,W] -> R dict(feats={idx: [B,K,C,h,w]}, v [B,K], gn [B,K]).
        메모리 feature 는 plain encoder(SCA/메모리 off) 출력, gradient 없음(추론 시 캐시 가능한 형태)."""
        B, K = mem.shape[:2]
        col = {}
        mx, mm = mem.flatten(0, 1), mem_m.flatten(0, 1)
        self._encode(torch.cat([mx, mm], 1), collect=col)
        feats = {i: f.view(B, K, *f.shape[1:]) for i, f in col.items()}
        v = mem_v.float()
        gn = (mem_gap.float().clamp(0, self.mem_gap) / self.mem_gap) * v
        R = {'feats': feats, 'v': v, 'gn': gn, 'gap': mem_gap.float()}
        if self.mem_mode == 'sam':                         # memory encoder 입력: 메모리 프레임에 대한 G 자신의 마스크 예측 (메모리 없이, eval BN)
            was = self.training
            self.eval()
            mh = torch.zeros_like(mm)
            idx = (v.flatten() > 0.5).nonzero().flatten()
            if len(idx):
                mh[idx] = self.forward(mx[idx], mm[idx])['m_eff'].to(mh.dtype)
            self.train(was)
            R['mask'] = mh.view(B, K, *mh.shape[1:])
        return R

    def forward(self, x, m, keep_sca=False, ref=None, ref_m=None, ref_v=None, ref_gap=None, ref_flow=None,
                mem=None, mem_m=None, mem_v=None, mem_gap=None, s2_bank=None, anchor_c=None):
        """anchor_c [B,2]: pass1(SCA prior) + 최종 창의 anchor 위치 강제 (None = 학습된 soft-argmax)."""
        self._s2_bank = s2_bank                            # s2m: 과거 프레임 memory bank (None/[] = 메모리 없음)
        anc_logit1 = None
        inp = torch.cat([x, m], 1)
        m_spec, sca_feats = None, None
        R = self.encode_ref(ref, ref_m, ref_v, ref_gap) if (self.ref_in and ref is not None) else None
        if self.mem_k and mem is not None:
            R = self.encode_mem(mem, mem_m, mem_v, mem_gap)
        if self.sca_idx:                                   # pass 1: target prior (box-prompt 에 해당)
            f1 = self._encode(inp, R=R)
            y1 = self._decode8(f1)
            if self.use_mask_head:                         # pass1 mask head 출력을 prior 로 (max 정규화)
                prior = self._mask_from_head(y1, x.shape[-2], x.shape[-1])
                prior = prior / (prior.amax((2, 3), keepdim=True) + 1e-6)
            elif self.anchor_r > 0 and self.sca_prior == 'anchor':
                prior, anchor1 = self._anchor_window(y1, x.shape[-2], x.shape[-1], anchor_c)
                anc_logit1 = self._anc_logit
            elif self.param in ('physical', 'hybrid'):
                w1 = self._anchor_window(y1, x.shape[-2], x.shape[-1])[0] if self.anchor_r > 0 else 1.0
                prior = torch.sigmoid(F.interpolate(self._mask_logit8(y1, f1[1], f1[0], inp), size=x.shape[-2:], mode='bilinear',
                                                    align_corners=False)) * w1
                prior = prior / (prior.amax((2, 3), keepdim=True) + 1e-6)
            else:
                prior = torch.ones_like(m)
            m_spec = m * prior
            sca_feats = [] if keep_sca else None
        feats = self._encode(inp, m_spec, sca_feats, R=R)
        s2, s4, s8, s16, s32 = feats
        y = self._decode8(feats)
        win = anchor = None
        phys8 = None
        if self.anchor_r > 0:
            win, anchor = self._anchor_window(y, x.shape[-2], x.shape[-1], anchor_c)
        if self.param in ('physical', 'hybrid'):
            p = self.head_phys(self._phys_feat(y, s4, s2, inp))
            if self.up_after:                              # /8 에서 비선형 적용 후 업샘플 -> 전이 폭 >= 1 셀(8px), 계단 경계 방지
                q = torch.cat([torch.sigmoid(p[:, 0:1]), torch.tanh(p[:, 1:2]), torch.sigmoid(p[:, 2:4])], 1)
                phys8 = q                                  # 평활 전 /8 맵 (map TV·포화 벌점용)
                if self.phys_blur > 0:
                    q = self._blur(q, self.pblur_k)
                q = F.interpolate(q, size=x.shape[-2:], mode='bilinear', align_corners=False)
                s_gain, t_off, s_alpha, m_hat = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
            else:                                          # (v1/v2) logit 업샘플 후 비선형 -> 큰 logit 이면 sub-cell 계단
                p = F.interpolate(p, size=x.shape[-2:], mode='bilinear', align_corners=False)
                s_gain, t_off, s_alpha, m_hat = (torch.sigmoid(p[:, 0:1]), torch.tanh(p[:, 1:2]),
                                                 torch.sigmoid(p[:, 2:3]), torch.sigmoid(p[:, 3:4]))
            gain = self.gain_lo + self.gain_span * s_gain
            off = self.max_offset * t_off
            alpha = self.alpha_lo + self.alpha_span * s_alpha   # <0: unsharp(선명화), >0: blur
            g = (x[:, :1] + 1) * 0.5
            out01 = gain * (g + alpha * (self._blur(g) - g)) + off
            if self.tex and ref is not None:              # 실제 VIS 참조 질감 이식: warp(ref) 의 HF + (LF - 현재 LF)
                out01 = out01 + self._tex(y, g, ref, ref_v, ref_flow)
            delta = 2 * (out01 - g)
            if self.param == 'hybrid':
                yd = self.up0(self.up1(self.up2(y, s4), s2), inp)
                detail = torch.tanh(self.head_detail(yd)) * self.detail_max
            delta = delta.expand(-1, x.shape[1], -1, -1)
        else:
            y = self.up2(y, s4)
            y = self.up1(y, s2)
            y = self.up0(y, inp)
            delta = torch.tanh(self.head_delta(y)) * self.max_delta
            m_hat = torch.sigmoid(self.head_mask(y))
        if self.use_mask_head:                             # head_phys mask 채널 대신 mask head, Gaussian 창 곱 없음
            m_hat, win = self._mask_from_head(y, x.shape[-2], x.shape[-1]), None
        if win is not None:
            m_hat = m_hat * win
        if self.param == 'hybrid':                         # detail_gate: 실효 가중 m·m̂² -> 경계보다 안쪽에서 소멸
            delta = delta + (detail * m_hat if self.detail_gate else detail).expand(-1, x.shape[1], -1, -1)
        if self.enhance_only:                              # 순방향 max(ΔI,0); 음수 영역은 0.1 기울기만 통과(죽은 gradient 방지)
            neg = (delta < 0).to(delta.dtype)
            delta = delta.clamp(min=0) + 0.1 * neg * (delta - delta.detach())
        m_eff = m * m_hat
        out = torch.clamp(x + delta * m_eff, -1.0, 1.0)     # I_out = I_vis + ΔI ⊙ M
        res = {'out': out, 'delta': delta, 'm_hat': m_hat, 'm_eff': m_eff, 'anchor': anchor,
               'anc_logit': getattr(self, '_anc_logit', None) if self.anchor_r > 0 else None, 'anc_hw': y.shape[-2:],
               'anc_logit1': anc_logit1, 'anchor1': anchor1 if anc_logit1 is not None else None,
               'm_spec': m_spec, 'sca_feats': sca_feats, 'phys8': phys8, 's2_pre': self._s2_pre}
        if R is not None and self.mem_mode == 'xattn':      # pass2 메모리 read-out (메모리 일관성 loss 용)
            res['mem_reads'] = {i: self.mem_fuse[str(i)].last_read for i in self.ref_idx}
            res['mem_valid'] = (R['v'].sum(1) > 0).float()
        return res

    def plain_feats(self, x, m):
        """SCA/메모리 off encoder 의 ref_idx 위치 feature (gradient 유지: I_out 재인코딩용)."""
        col = {}
        self._encode(torch.cat([x, m], 1), collect=col)
        return col


class DeployGenerator(nn.Module):
    """export 용: (I_vis, M) -> I_out 만 반환."""
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x, m):
        return self.g(x, m)['out']
