"""COIN counterfactual inpainting adapted to femoral nerve ultrasound: ANISOTROPIC -> NERVE_VISIBLE.

Differences to `CounterfactualInpaintingCGANV2`:
  * black-box classifier is the frozen 3-class TBPTT net (ConvNeXt-T + ConvGRU) that scores a *window* of
    S consecutive frames; batches are [B,S,1,H,W] gray windows and the generator works per frame (N=B*S).
  * per-frame condition c = 1 (inpaint) iff argmax f(x) == ANISOTROPIC, else 0 (identity).
    NERVE_NOT_VISIBLE frames are simply "keep" frames — they are never pushed anywhere.
  * classifier consistency: CE(f(x_cf), NERVE_VISIBLE) on inpainted frames + KL(f(x) || f(x_cf)) on kept frames.
  * the generator emits only the gray frame; the classifier's [gray, CLAHE, scharr] input is re-derived
    from it (see `tbptt_classifier.derive_channels`), so the counterfactual is a real image, not 3 free maps.
  * discriminator is conditioned on the classifier's 3-way argmax; fakes are conditioned on the desired
    class (VISIBLE for inpainted frames, own class for kept frames).
  * optional temporal ConvGRU at the encoder bottleneck + temporal smoothness of the perturbation, and an
    EMA copy of E/G used for evaluation.
  * optional *visible-frame memory* (`vis_memory`): a space-time-memory read (Oh et al. 2019) at selected
    encoder levels whose keys/values are encoder features of the P past context frames + earlier frames of
    the window, restricted to frames the classifier calls NERVE_VISIBLE (hard mask on argmax, log p_vis bias).
    So an ANISOTROPIC frame can copy how the nerve looked the last time it was visible instead of hallucinating it.
No lightning Fabric: plain fp32 training with autocast only inside the frozen classifier.
"""
import copy
import itertools
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import loss_hinge_dis, loss_hinge_gen, tv_loss
from src.models.cgan.discriminator import ResBlocksDiscriminator
from src.models.cgan.generator import ResBlocksEncoder, ResBlocksGenerator
from src.models.tbptt_classifier import CLASSES, ConvGRUCell, derive_channels, load_tbptt_classifier


class TemporalBottleneck(nn.Module):
    """ConvGRU over the deepest encoder feature map (causal within the window), fused residually."""

    def __init__(self, channels):
        super().__init__()
        self.cell = ConvGRUCell(channels, channels, 3)
        self.fuse = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, z, B, S):
        zs = z.reshape(B, S, *z.shape[1:])
        h = torch.zeros_like(zs[:, 0])
        outs = []
        for t in range(S):
            h = self.cell(zs[:, t], h)
            outs.append(h)
        h = torch.stack(outs, 1).reshape_as(z)
        return z + self.fuse(h)


class VisMemoryRead(nn.Module):
    """Space-time memory read at one encoder level.

    Queries: window features z [B*S,C,h,w]. Memory: features of T = P (context) + S (window) frames [B,T,C,h,w]
    with per-frame weight w [B,T] (0 = unreadable). Frame s of the window attends to memory frames t < P + s
    (strictly past) that have w > 0, with a log(w) bias, over all (pooled) spatial positions; a learned null slot
    keeps the softmax defined when no visible frame has been seen yet. Fused residually with a zero-init 1x1.
    """

    def __init__(self, channels, key_dim=64, val_dim=None, pool=1):
        super().__init__()
        val_dim = val_dim or channels // 4
        self.q = nn.Conv2d(channels, key_dim, 1)
        self.k = nn.Conv2d(channels, key_dim, 1)
        self.v = nn.Conv2d(channels, val_dim, 1)
        self.null_v = nn.Parameter(torch.zeros(val_dim))
        self.fuse = nn.Conv2d(val_dim, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.pool, self.key_dim = pool, key_dim

    def forward(self, z, mem, w, B, S):
        T = mem.shape[1]
        P = T - S
        C, h, wd = z.shape[1:]
        m = mem.reshape(B * T, C, h, wd)
        if self.pool > 1:
            m = F.avg_pool2d(m, self.pool)
        k = self.k(m).reshape(B, T, self.key_dim, -1)        # [B,T,dk,L]
        v = self.v(m).reshape(B, T, -1, k.shape[-1])          # [B,T,dv,L]
        q = self.q(z).reshape(B, S, self.key_dim, -1)         # [B,S,dk,Lq]
        logits = torch.einsum('bsdq,btdl->bsqtl', q, k) / self.key_dim ** 0.5
        causal = torch.arange(T, device=z.device)[None, :] < (P + torch.arange(S, device=z.device))[:, None]  # [S,T]
        readable = causal[None] & (w > 0)[:, None, :]                                                        # [B,S,T]
        logits = logits + torch.log(w.clamp_min(1e-6))[:, None, None, :, None]
        logits = logits.masked_fill(~readable[:, :, None, :, None], float('-inf'))
        L, Lq = k.shape[-1], q.shape[-1]
        logits = torch.cat([logits.reshape(B, S, Lq, T * L), logits.new_zeros(B, S, Lq, 1)], -1)  # + null slot
        attn = logits.softmax(-1)
        vflat = v.permute(0, 1, 3, 2).reshape(B, T * L, -1)                                        # [B,T*L,dv]
        read = torch.einsum('bsqm,bmd->bsqd', attn[..., :-1], vflat) + attn[..., -1:] * self.null_v
        read = read.permute(0, 1, 3, 2).reshape(B * S, -1, h, wd)
        return z + self.fuse(read)


def _phase_corr_shift(a, b, max_shift=40):
    """Translational registration via FFT phase correlation. a,b: [N,1,H,W] in [-1,1] (same N).
    Returns integer per-sample (dy,dx) pixel shift that best aligns `a` onto `b`, clamped to `max_shift`
    (US frames are noisy/low-contrast, so a large "peak" is more likely spurious aliasing than real drift)."""
    eps = 1e-6
    Fa = torch.fft.fft2(a.float())
    Fb = torch.fft.fft2(b.float())
    R = Fa * Fb.conj()
    R = R / (R.abs() + eps)
    r = torch.fft.ifft2(R).real  # cross-power spectrum -> correlation surface, peak at the aligning shift
    N, _, H, W = r.shape
    idx = r.reshape(N, -1).argmax(-1)
    dy, dx = idx // W, idx % W
    dy = torch.where(dy > H // 2, dy - H, dy)
    dx = torch.where(dx > W // 2, dx - W, dx)
    return dy.clamp(-max_shift, max_shift), dx.clamp(-max_shift, max_shift)


def _warp_translate(a, dy, dx):
    """Zero-padded per-sample translation. a: [N,1,H,W]; dy,dx: [N] pixel shifts (+dy = content moves down)."""
    N, C, H, W = a.shape
    theta = a.new_zeros(N, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2.0 * dx / W
    theta[:, 1, 2] = -2.0 * dy / H
    grid = F.affine_grid(theta, a.shape, align_corners=False)
    return F.grid_sample(a, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def align_to_reference(frames, ref):
    """CoSpliced-style anchor alignment (VidSplice): register each memory-bank frame onto a common reference
    (here, the most recent frame the model has actually seen) before it is used as a read target, so that
    probe drift between past NERVE_VISIBLE frames doesn't smear the memory read. frames,ref: [N,1,H,W].
    `_phase_corr_shift(frames, ref)` returns the shift already present in `frames` relative to `ref` (i.e. the
    shift that was applied to produce `frames` from `ref`-like content), so the correction is its negation."""
    dy, dx = _phase_corr_shift(frames, ref)
    return _warp_translate(frames, -dy.float(), -dx.float())


class VisMemory(nn.Module):
    """One `VisMemoryRead` per selected encoder level (index into the encoder feature list)."""

    def __init__(self, enc_channels, levels=(3, 4), key_dim=64, pool=None):
        super().__init__()
        pool = {int(l): int(f) for l, f in (pool or [])}  # [[level, factor], ...] (yaml/EasyDict cannot hold int keys)
        self.levels = list(levels)
        self.reads = nn.ModuleDict({str(l): VisMemoryRead(enc_channels[l], key_dim, pool=pool.get(l, 1)) for l in self.levels})

    def forward(self, feats, mem, w, B, S):
        return [self.reads[str(l)](f, mem[l], w, B, S) if l in self.levels else f for l, f in enumerate(feats)]


class CounterfactualInpaintingCGANFemoral(nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opt = opt
        self.img_size = img_size
        self.source_class_idx = opt.get('source_class_idx', CLASSES.index('ANISOTROPIC'))
        self.target_class_idx = opt.get('target_class_idx', CLASSES.index('NERVE_VISIBLE'))
        self.n_classes = len(CLASSES)
        self.num_bins = 2  # generator condition: 0 = identity, 1 = inpaint
        self.ptb_based = opt.get('ptb_based', True)

        self.enc = ResBlocksEncoder(1, **opt.get('enc_params', {}))
        self.gen = ResBlocksGenerator(self.num_bins, in_channels=self.enc.out_channels, **opt.get('gen_params', {}))
        self.disc = ResBlocksDiscriminator(self.n_classes, 1, **opt.get('disc_params', {}))
        self.tbn = TemporalBottleneck(self.enc.out_channels[-1]) if opt.get('temporal_bottleneck', True) else None
        vm = opt.get('vis_memory')
        self.mem = VisMemory(self.enc.out_channels, vm.get('levels', [3, 4]), vm.get('key_dim', 64), vm.get('pool')) if vm else None
        self.mem_thresh = (vm or {}).get('thresh', 0.5)  # a frame is readable iff argmax == VISIBLE and p_vis >= thresh
        self.mem_align = (vm or {}).get('align', False)  # VidSplice/CoSpliced-style: register context frames onto the current frame before reading

        self.classifier_f = load_tbptt_classifier(opt.classifier_ckpt, device='cpu')
        self.clf_autocast = opt.get('clf_autocast', True)

        self.adv_loss = opt.get('adv_loss', 'mse')
        self.adversarial_loss = nn.MSELoss() if self.adv_loss == 'mse' else nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.lambda_adv = opt.get('lambda_adv', 1.0)
        self.lambda_cls = opt.get('lambda_cls', 1.0)     # CE -> target class on inpainted frames
        self.lambda_keep = opt.get('lambda_keep', 1.0)   # KL keep on non-source frames
        self.lambda_rec = opt.get('lambda_rec', 1.0)
        self.lambda_tv = opt.get('lambda_tv', 0.0)
        self.lambda_ptb = opt.get('lambda_ptb', 0.0)     # L1 sparsity of the perturbation
        self.lambda_temp = opt.get('lambda_temp', 0.0)   # |delta_t - delta_{t-1}|
        self.cyclic_rec = opt.get('cyclic_rec', True)
        self.gen_update_freq = opt.get('gen_update_freq', 1)
        self.eps = opt.get('eps', 1e-8)

        gen_params = [self.enc.parameters(), self.gen.parameters()] + [m.parameters() for m in (self.tbn, self.mem) if m is not None]
        self.optimizer_G = torch.optim.Adam(itertools.chain(*gen_params), lr=opt.lr, betas=(opt.b1, opt.b2), eps=self.eps)
        self.optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=self.eps)

        self.ema_decay = opt.get('ema_decay', 0.0)
        self.ema = None
        if self.ema_decay > 0:
            self.ema = nn.ModuleDict({'enc': copy.deepcopy(self.enc), 'gen': copy.deepcopy(self.gen)})
            for k in ('tbn', 'mem'):
                if getattr(self, k) is not None:
                    self.ema[k] = copy.deepcopy(getattr(self, k))
            self.ema.requires_grad_(False)
        self.gen_loss_logs, self.disc_loss_logs = {}, {}

    # ------------------------------------------------------------------ helpers
    def train(self, mode: bool = True):
        ret = super().train(mode)
        self.classifier_f.eval()
        if self.ema is not None:
            self.ema.eval()
        return ret

    def state_dict(self, *args, **kwargs):
        # the frozen classifier lives in its own checkpoint; keep ours small
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith('classifier_f.')}

    def load_state_dict(self, state_dict, strict=True):
        clf = {k: v for k, v in self.classifier_f.state_dict().items()}
        merged = dict(state_dict)
        merged.update({f'classifier_f.{k}': v for k, v in clf.items()})
        return super().load_state_dict(merged, strict=strict)

    @torch.no_grad()
    def update_ema(self):
        if self.ema is None:
            return
        pairs = [(self.ema[k], getattr(self, k)) for k in self.ema]
        for e, m in pairs:
            for pe, pm in zip(e.parameters(), m.parameters()):
                pe.lerp_(pm, 1 - self.ema_decay)
            for be, bm in zip(e.buffers(), m.buffers()):
                be.copy_(bm)

    @contextmanager
    def ema_scope(self):
        """Temporarily swap E/G(/tbn) weights with their EMA copies."""
        if self.ema is None:
            yield
            return
        backup = {k: copy.deepcopy(getattr(self, k).state_dict()) for k in self.ema}
        for k in self.ema:
            getattr(self, k).load_state_dict(self.ema[k].state_dict())
        try:
            yield
        finally:
            for k in self.ema:
                getattr(self, k).load_state_dict(backup[k])

    def classify(self, gray, B, S):
        """gray [N,1,H,W] in [-1,1] -> per-frame logits [N,3] (window-aware, grads flow to gray)."""
        x3 = derive_channels(gray)
        x3 = x3.reshape(B, S, *x3.shape[1:])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.clf_autocast):
            logits = self.classifier_f(x3)
        return logits.float().reshape(B * S, -1)

    @torch.no_grad()
    def condition(self, gray, B, S):
        """Classifier-driven condition per frame."""
        probs = self.classify(gray, B, S).softmax(-1)
        pred = probs.argmax(-1)
        c = (pred == self.source_class_idx).long()
        desired = torch.where(c.bool(), torch.full_like(pred, self.target_class_idx), pred)
        return probs, pred, c, desired

    def vis_weight(self, probs):
        """[N,3] classifier probs -> [N] memory weight: p_vis on frames called VISIBLE (>= thresh), else 0."""
        p = probs[:, self.target_class_idx]
        ok = (probs.argmax(-1) == self.target_class_idx) & (p >= self.mem_thresh)
        return p * ok.float()

    @torch.no_grad()
    def prepare_context(self, ctx, ctx_valid, probs_x, B, S, ref=None):
        """ctx [B,P,1,H,W] past frames -> dict(feats={level: [B,P,C,h,w]}, w=[B,P+S]) shared by all generator passes.
        Context features are encoder-only, no grad (the memory is read, not trained through the past).
        `ref` [B,1,H,W]: the most recent frame the model has seen (last window frame); if `mem_align`, every
        context frame is first registered onto it (translation only, FFT phase correlation) so that probe
        drift across the (up to `context` frames back) memory bank doesn't smear the space-time-memory read."""
        if self.mem is None:
            return None
        w_win = self.vis_weight(probs_x).reshape(B, S)
        if ctx is None or ctx.shape[1] == 0:
            return {'feats': {l: None for l in self.mem.levels}, 'w': w_win}
        P = ctx.shape[1]
        assert P % S == 0, f'context {P} must be a multiple of seq {S} (classifier hidden reset per window)'
        flat = ctx.reshape(B * P, *ctx.shape[2:])
        if self.mem_align and ref is not None:
            flat = align_to_reference(flat, ref.repeat_interleave(P, dim=0))
        feats = self.enc(flat)
        probs = self.classify(flat, B * P // S, S).softmax(-1)  # [B*P,3]
        w_ctx = self.vis_weight(probs).reshape(B, P) * ctx_valid.to(probs.dtype)
        return {'feats': {l: feats[l].reshape(B, P, *feats[l].shape[1:]) for l in self.mem.levels},
                'w': torch.cat([w_ctx, w_win], 1)}

    def explanation_function(self, gray, c, B, S, ctx=None, win_feats=None, ret_feats=False):
        """ctx: output of `prepare_context`; win_feats: raw window features to use as in-window memory instead of
        the current ones (cyclic pass reads the memory built from the original x)."""
        feats = self.enc(gray)
        raw = feats
        if self.tbn is not None:
            feats = feats[:-1] + [self.tbn(feats[-1], B, S)]
        if self.mem is not None:
            assert ctx is not None, 'vis_memory needs the past-frame context (dataset.context > 0)'
            src = win_feats or raw
            mem = {}
            for l in self.mem.levels:
                win = src[l].reshape(B, S, *src[l].shape[1:])
                mem[l] = win if ctx['feats'][l] is None else torch.cat([ctx['feats'][l], win], 1)
            feats = self.mem(feats, mem, ctx['w'], B, S)
        x_cf = self.gen(feats, c, x=gray if self.ptb_based else None)
        return (x_cf, raw) if ret_feats else x_cf

    @torch.no_grad()
    def counterfactual(self, gray, ctx=None, ctx_valid=None):
        """Inference: gray [B,S,1,H,W] (+ ctx [B,P,1,H,W], ctx_valid [B,P]) -> (x_cf, probs_x, probs_cf, c)."""
        B, S = gray.shape[:2]
        flat = gray.reshape(B * S, *gray.shape[2:])
        probs, pred, c, _ = self.condition(flat, B, S)
        with self.ema_scope():
            ctx_info = self.prepare_context(ctx, ctx_valid, probs, B, S, ref=gray[:, -1])
            x_cf = self.explanation_function(flat, c, B, S, ctx_info)
        probs_cf = self.classify(x_cf, B, S).softmax(-1)
        v = lambda t: t.reshape(B, S, *t.shape[1:])
        return v(x_cf), v(probs), v(probs_cf), v(c)

    # ------------------------------------------------------------------ training step
    def forward(self, batch, training=False, validation=False, compute_norms=False, global_step=None):
        assert training != validation
        imgs = batch['image'].cuda(non_blocking=True)  # [B,S,1,H,W] in [-1,1]
        B, S = imgs.shape[:2]
        N = B * S
        x = imgs.reshape(N, *imgs.shape[2:])
        valid = x.new_ones(N, 1)
        fake = x.new_zeros(N, 1)

        probs_x, pred_x, c, desired = self.condition(x, B, S)
        inpaint = c.bool()
        keep = ~inpaint
        ctx = None
        if self.mem is not None:
            ctx = self.prepare_context(batch['ctx'].cuda(non_blocking=True), batch['ctx_valid'].cuda(non_blocking=True), probs_x, B, S, ref=imgs[:, -1])

        # ----------------------------------------------------------- generator
        if training:
            self.optimizer_G.zero_grad(set_to_none=True)
        x_cf, raw_feats = self.explanation_function(x, c, B, S, ctx, ret_feats=True)
        delta = x_cf - x

        update_generator = global_step is not None and global_step % self.gen_update_freq == 0
        if update_generator or validation:
            dis_fake = self.disc(x_cf, desired)
            g_adv = self.lambda_adv * (loss_hinge_gen(dis_fake) if self.adv_loss == 'hinge' else self.adversarial_loss(dis_fake, valid))

            logits_cf = self.classify(x_cf, B, S)
            logp_cf = F.log_softmax(logits_cf, -1)
            if inpaint.any():
                g_cls = self.lambda_cls * F.nll_loss(logp_cf[inpaint], torch.full((int(inpaint.sum()),), self.target_class_idx, device=x.device, dtype=torch.long))
            else:
                g_cls = x.new_zeros(())
            if keep.any():
                g_keep = self.lambda_keep * F.kl_div(logp_cf[keep], probs_x[keep], reduction='batchmean')
            else:
                g_keep = x.new_zeros(())

            g_rec = self.lambda_rec * self.l1(x_cf, x)
            if self.cyclic_rec and self.lambda_rec > 0:
                # I(I(x, c), 0) ~= I(x, c): a counterfactual is a fixed point of the identity condition
                x_cyc = self.explanation_function(x_cf, torch.zeros_like(c), B, S, ctx, win_feats=[f.detach() for f in raw_feats])
                g_rec = g_rec + self.lambda_rec * self.l1(x_cyc, x_cf)

            g_tv = self.lambda_tv * tv_loss(delta.abs().mul(127.5)) if self.lambda_tv else x.new_zeros(())
            g_ptb = self.lambda_ptb * delta.abs().mean() if self.lambda_ptb else x.new_zeros(())
            if self.lambda_temp and S > 1:
                d = delta.reshape(B, S, *delta.shape[1:])
                g_temp = self.lambda_temp * (d[:, 1:] - d[:, :-1]).abs().mean()
            else:
                g_temp = x.new_zeros(())

            g_loss = g_adv + g_cls + g_keep + g_rec + g_tv + g_ptb + g_temp
            if update_generator:
                g_loss.backward()
                self.optimizer_G.step()
                self.update_ema()

            with torch.no_grad():
                pred_cf = logits_cf.argmax(-1)
                flip = (pred_cf[inpaint] == self.target_class_idx).float().mean().item() if inpaint.any() else float('nan')
                kept = (pred_cf[keep] == pred_x[keep]).float().mean().item() if keep.any() else float('nan')
            self.gen_loss_logs.update(g_adv=g_adv.item(), g_cls=g_cls.item(), g_keep=g_keep.item(), g_rec=g_rec.item(),
                                      g_tv=g_tv.item(), g_ptb=g_ptb.item(), g_temp=g_temp.item(), g_loss=g_loss.item(),
                                      flip_rate=flip, keep_rate=kept, frac_inpaint=inpaint.float().mean().item())

        # ----------------------------------------------------------- discriminator
        if training:
            self.optimizer_D.zero_grad(set_to_none=True)
        dis_real = self.disc(x, pred_x)
        dis_fake = self.disc(x_cf.detach(), desired)
        if self.adv_loss == 'hinge':
            d_real, d_fake = loss_hinge_dis(dis_fake, dis_real)
        else:
            d_real, d_fake = self.adversarial_loss(dis_real, valid), self.adversarial_loss(dis_fake, fake)
        d_loss = (d_real + d_fake) / 2
        if training:
            d_loss.backward()
            self.optimizer_D.step()
        self.disc_loss_logs.update(d_real_loss=d_real.item(), d_fake_loss=d_fake.item(), d_loss=d_loss.item())

        return {
            'loss': {**self.gen_loss_logs, **self.disc_loss_logs},
            'gen_imgs': x_cf.detach(),
            'cond': c,
        }
