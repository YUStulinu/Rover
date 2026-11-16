"""A small U-Net with two heads: depth (meters, through its logarithm) and
the kind of surface of every pixel (floor, wall, ceiling, obstacle), from
one grayscale image. Trained on the simulator only."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    MIN_DEPTH, MAX_DEPTH = 0.1, 20.0

    def __init__(self, base=24, classes=4):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([block(1, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3]), block(c[3], c[4])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in range(4)])
        self.depth = nn.Conv2d(c[0], 1, 1)
        self.seg = nn.Conv2d(c[0], classes, 1)

    def forward(self, x):
        # pad to a multiple of 16
        H, W = x.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for i in reversed(range(4)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        lo, hi = math.log(self.MIN_DEPTH), math.log(self.MAX_DEPTH)
        log_depth = lo + (hi - lo) * torch.sigmoid(self.depth(x))
        return log_depth[..., :H, :W], self.seg(x)[..., :H, :W]


def silog_loss(log_pred, depth, valid, lam=0.85):
    d = (log_pred - torch.log(depth.clamp(min=1e-3)))[valid]
    if d.numel() == 0:
        return log_pred.sum() * 0
    return torch.sqrt((d**2).mean() - lam * d.mean() ** 2 + 1e-8) + 0.5 * d.abs().mean()
