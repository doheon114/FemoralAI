"""Trainer for the femoral ANISOTROPIC -> NERVE_VISIBLE counterfactual inpainting GAN.

There are no GT masks for the nerve, so evaluation is classifier-driven:
    flip_pred : P(argmax f(x_cf) == VISIBLE | argmax f(x) == ANISO)      (counterfactual validity)
    flip_gt   : P(argmax f(x_cf) == VISIBLE | GT == ANISO)
    keep      : P(argmax f(x_cf) == argmax f(x) | argmax f(x) != ANISO)  (identity frames untouched)
    l1_ani/l1_keep : mean |x_cf - x| (in [0,1] units) on inpainted / kept frames  (minimality)
    score     = flip_pred + keep - l1_ani   -> `best.pth`
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.femoral import CLASSES, FemoralWindowDataset
from src.trainers.trainer import BaseTrainer
from src.utils.avg_meter import AvgMeter

SHORT = {'ANISOTROPIC': 'ANI', 'NERVE_NOT_VISIBLE': 'ABS', 'NERVE_VISIBLE': 'VIS'}


class CounterfactualInpaintingFemoralTrainer(BaseTrainer):
    def __init__(self, opt: edict, model, continue_path: str = None) -> None:
        super().__init__(opt, model, continue_path)
        self.cf_vis_dir = self.logging_dir / 'counterfactuals'
        self.cf_vis_dir.mkdir(exist_ok=True)
        self.best_score = -math.inf
        self.n_vis = opt.get('n_vis_windows', 6)

    # ------------------------------------------------------------------ state
    def restore_state(self):
        ckpt = self.ckpt_dir / (self.ckpt_name or 'latest.pth')
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.batches_done = state['step']
        self.current_epoch = state['epoch'] + 1
        self.best_score = state.get('best_score', -math.inf)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer_G.load_state_dict(state['optimizers'][0])
        self.model.optimizer_D.load_state_dict(state['optimizers'][1])
        self.logger.info(f'Restored checkpoint {ckpt} (epoch {state["epoch"]})')

    def save_state(self, name='latest.pth') -> str:
        state = {
            'config': self.opt, 'model': self.model.state_dict(),
            'step': self.batches_done, 'epoch': self.current_epoch, 'best_score': self.best_score,
        }
        if name == 'latest.pth':  # best.pth is for inference only -> no optimizer state (halves the size)
            state['optimizers'] = [self.model.optimizer_G.state_dict(), self.model.optimizer_D.state_dict()]
        path = self.ckpt_dir / name
        torch.save(state, path)
        return str(path)

    # ------------------------------------------------------------------ data
    def get_dataloaders(self):
        d = self.opt.dataset
        common = dict(root_dir=d.root_dir, seq=d.seq, img_size=d.img_size[0], context=d.get('context', 0))
        train_ds = FemoralWindowDataset(split=d.get('train_split', 'train'), stride=d.get('train_stride', d.seq // 2),
                                        hflip=d.get('hflip', True), limit=d.get('limit_train'), **common)
        val_ds = FemoralWindowDataset(split=d.get('val_split', 'val'), stride=d.seq, limit=d.get('limit_val'), **common)
        self.logger.info(f'train windows={len(train_ds)} (videos={train_ds.n_videos}) frames/class={train_ds.label_counts().tolist()}')
        self.logger.info(f'val   windows={len(val_ds)} (videos={val_ds.n_videos}) frames/class={val_ds.label_counts().tolist()}')
        train_loader = DataLoader(train_ds, batch_size=d.batch_size, shuffle=True, drop_last=True,
                                  num_workers=d.num_workers, pin_memory=True, persistent_workers=d.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=d.get('val_batch_size', d.batch_size), shuffle=False,
                                num_workers=d.num_workers, pin_memory=True, persistent_workers=d.num_workers > 0)
        return train_loader, val_loader

    # ------------------------------------------------------------------ loops
    def training_epoch(self, loader):
        self.model.train()
        stats = AvgMeter()
        epoch_steps = self.opt.get('epoch_steps')
        with tqdm(enumerate(loader), desc=f'Training epoch {self.current_epoch}', leave=False, total=epoch_steps or len(loader)) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                self.batches_done = self.current_epoch * len(loader) + i
                outs = self.model(batch, training=True, global_step=self.batches_done)
                stats.update({k: v for k, v in outs['loss'].items() if not (isinstance(v, float) and math.isnan(v))})
                if self.batches_done % self.opt.sample_interval == 0:
                    l = outs['loss']
                    prog.set_postfix_str('[D %.3f] [G %.3f cls %.3f keep %.3f rec %.3f] [flip %.2f keep %.2f]' % (
                        l['d_loss'], l['g_loss'], l['g_cls'], l['g_keep'], l['g_rec'], l.get('flip_rate', 0), l.get('keep_rate', 0)), refresh=True)
        epoch_stats = stats.average()
        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info('[train %d/%d] D %.4f | G %.4f (adv %.3f cls %.3f keep %.3f rec %.3f tv %.3f ptb %.3f temp %.3f) | flip %.3f keep %.3f inpaint-frac %.3f' % (
            self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'], epoch_stats['g_adv'], epoch_stats['g_cls'],
            epoch_stats['g_keep'], epoch_stats['g_rec'], epoch_stats['g_tv'], epoch_stats['g_ptb'], epoch_stats['g_temp'],
            epoch_stats.get('flip_rate', float('nan')), epoch_stats.get('keep_rate', float('nan')), epoch_stats['frac_inpaint']))
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader):
        self.model.eval()
        metrics = self.evaluate_counterfactual(loader)
        self.logger.log(metrics, self.current_epoch, 'val')
        self.logger.info('[val   %d/%d] flip_pred %.3f flip_gt %.3f keep %.3f | p_vis(ANI) %.3f->%.3f | l1_ani %.4f l1_keep %.4f | score %.4f' % (
            self.current_epoch, self.opt.n_epochs, metrics['flip_pred'], metrics['flip_gt'], metrics['keep'],
            metrics['pvis_ani_before'], metrics['pvis_ani_after'], metrics['l1_ani'], metrics['l1_keep'], metrics['score']))
        if metrics['score'] > self.best_score:
            self.best_score = metrics['score']
            self.logger.info(f'New best score {self.best_score:.4f} -> {self.save_state("best.pth")}')
        return metrics

    def fit(self):
        train_loader, val_loader = self.get_dataloaders()
        for _ in range(self.current_epoch, self.opt.n_epochs):
            self.training_epoch(train_loader)
            self.validation_epoch(val_loader)
            self.logger.info(f'Saved {self.save_state()}')
            self.current_epoch += 1

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate_counterfactual(self, loader):
        self.model.eval()
        src, tgt = self.model.source_class_idx, self.model.target_class_idx
        n_flip_p = n_pred_src = n_flip_g = n_gt_src = n_keep = n_keep_tot = 0
        l1_ani = l1_keep = 0.0
        pvis_b = pvis_a = 0.0
        n_vis = 0
        for i, batch in tqdm(enumerate(loader), desc='Evaluating counterfactuals', leave=False, total=len(loader)):
            x = batch['image'].cuda(non_blocking=True)
            y = batch['label'].cuda()
            ctx = batch['ctx'].cuda(non_blocking=True) if 'ctx' in batch else None
            x_cf, p_x, p_cf, c = self.model.counterfactual(x, ctx, batch['ctx_valid'].cuda() if ctx is not None else None)
            pred_x, pred_cf = p_x.argmax(-1), p_cf.argmax(-1)
            inpaint, keep = c.bool(), ~c.bool()
            gt_src = y == src
            n_pred_src += inpaint.sum().item()
            n_flip_p += (pred_cf[inpaint] == tgt).sum().item()
            n_gt_src += gt_src.sum().item()
            n_flip_g += (pred_cf[gt_src] == tgt).sum().item()
            n_keep_tot += keep.sum().item()
            n_keep += (pred_cf[keep] == pred_x[keep]).sum().item()
            d = (x_cf - x).abs().mean(dim=(2, 3, 4)) / 2  # [B,S], [0,1] units
            l1_ani += d[inpaint].sum().item()
            l1_keep += d[keep].sum().item()
            pvis_b += p_x[inpaint][:, tgt].sum().item()
            pvis_a += p_cf[inpaint][:, tgt].sum().item()
            if n_vis < self.n_vis and inpaint.any():
                b = int(inpaint.any(dim=1).nonzero()[0])
                self.visualize_window(x[b], x_cf[b], y[b], p_x[b], p_cf[b], c[b],
                                      self.cf_vis_dir / f'epoch_{self.current_epoch:03d}_{batch["vid"][b]}_s{int(batch["start"][b])}.png')
                n_vis += 1
        m = {
            'flip_pred': n_flip_p / max(n_pred_src, 1), 'flip_gt': n_flip_g / max(n_gt_src, 1),
            'keep': n_keep / max(n_keep_tot, 1),
            'l1_ani': l1_ani / max(n_pred_src, 1), 'l1_keep': l1_keep / max(n_keep_tot, 1),
            'pvis_ani_before': pvis_b / max(n_pred_src, 1), 'pvis_ani_after': pvis_a / max(n_pred_src, 1),
            'n_pred_src': n_pred_src, 'n_gt_src': n_gt_src,
        }
        m['score'] = m['flip_pred'] + m['keep'] - m['l1_ani']
        return m

    @staticmethod
    def visualize_window(x, x_cf, y, p_x, p_cf, c, path: Path):
        """x, x_cf: [S,1,H,W] in [-1,1]. Rows: input / counterfactual / |diff|; one column per frame."""
        S = x.shape[0]
        x, x_cf = (x[:, 0].cpu().numpy() + 1) / 2, (x_cf[:, 0].cpu().numpy() + 1) / 2
        diff = np.abs(x_cf - x)
        fig, axes = plt.subplots(3, S, figsize=(1.9 * S, 6.2))
        for t in range(S):
            gt = SHORT[CLASSES[int(y[t])]]
            fx, fcf = p_x[t].cpu().numpy(), p_cf[t].cpu().numpy()
            axes[0, t].imshow(x[t], cmap='gray', vmin=0, vmax=1)
            axes[0, t].set_title(f'GT {gt} | f(x) {SHORT[CLASSES[int(fx.argmax())]]} {fx.max():.2f}' + (' *' if c[t] else ''), fontsize=7)
            axes[1, t].imshow(x_cf[t], cmap='gray', vmin=0, vmax=1)
            axes[1, t].set_title(f'f(x_cf) {SHORT[CLASSES[int(fcf.argmax())]]} {fcf.max():.2f} | pVIS {fcf[2]:.2f}', fontsize=7)
            axes[2, t].imshow(diff[t], cmap='inferno', vmin=0, vmax=0.5)
            axes[2, t].set_title(f'|d| mean {diff[t].mean():.3f}', fontsize=7)
            for r in range(3):
                axes[r, t].axis('off')
        fig.suptitle(f'{path.stem}   (* = inpainted frame, condition ANI->VIS)', fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=80)
        plt.close(fig)
