"""Lightweight 70x70 PatchGAN Discriminator (pix2pix 구조, n_layers=3 -> receptive field 70).

기본 입력은 gray 1채널: 생성 이미지는 CLAHE/scharr 를 gray 에서 재유도하므로, 3채널을 보면
판별기가 재유도 채널의 미세한 계산 차이(비본질적 단서)로 real/fake 를 구분할 수 있기 때문.
"""
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, ndf=32, n_layers=3, use_sn=True):
        super().__init__()
        sn = spectral_norm if use_sn else (lambda m: m)
        layers = [sn(nn.Conv2d(in_ch, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):                        # stride 2 x (n_layers-1)
            prev, mult = mult, min(2 ** n, 8)
            layers += [sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)),
                       nn.InstanceNorm2d(ndf * mult, affine=True), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)            # stride 1
        layers += [sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)),
                   nn.InstanceNorm2d(ndf * mult, affine=True), nn.LeakyReLU(0.2, True),
                   sn(nn.Conv2d(ndf * mult, 1, 4, 1, 1))]   # patch logits [B,1,h',w'] (224 -> 26x26)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
