"""Stereo geometry and classical stereo depth.

- Rectification (Fusiello, Trucco and Verri 2000): both cameras are turned
  to look along the same direction with the x axis along the baseline, and
  the images are resampled (undistorting EuRoC's radial-tangential lens), so
  that matching points lie on the same row.
- Semi-global matching (Hirschmuller 2008) on census-transform costs, with
  four aggregation directions, sub-pixel refinement and a uniqueness check,
  vectorized in PyTorch (GPU when available).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import Camera


class Rectifier:
    def __init__(self, c0: Camera, c1: Camera, size=(376, 240), f=230.0, row_offset=0.0):
        T_c1c0 = np.linalg.inv(c1.T_bc) @ c0.T_bc
        R, t = T_c1c0[:3, :3], T_c1c0[:3, 3]
        c1_in_0 = -R.T @ t  # the right camera's center in the left frame
        r1 = c1_in_0 / np.linalg.norm(c1_in_0)
        z0 = np.array([0, 0, 1.0])
        r2 = np.cross(z0, r1)
        r2 /= np.linalg.norm(r2)
        r3 = np.cross(r1, r2)
        self.R0 = np.stack([r1, r2, r3])  # left camera -> rectified
        self.R1 = self.R0 @ R.T  # right camera -> rectified
        self.baseline = float(np.linalg.norm(c1_in_0))
        w, h = size
        self.size = size
        self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        self.row_offset = row_offset  # rows added to the right image's sampling (rig misalignment)
        self.maps = [self._map(c0, self.R0), self._map(c1, self.R1, row_offset)]

    def _map(self, cam: Camera, Rr: np.ndarray, dv=0.0):
        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float) + dv)
        rays = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
        cam_rays = rays @ Rr  # rows: Rr^T applied to each ray
        x, y = cam_rays[..., 0] / cam_rays[..., 2], cam_rays[..., 1] / cam_rays[..., 2]
        xd, yd = cam.distort(x, y)
        px = cam.K[0, 0] * xd + cam.K[0, 2]
        py = cam.K[1, 1] * yd + cam.K[1, 2]
        W, H = cam.size
        # normalized coordinates for grid_sample
        grid = np.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], -1)
        return torch.tensor(grid, dtype=torch.float32)[None]

    def rectify(self, img: np.ndarray, k: int) -> np.ndarray:
        t = torch.tensor(img, dtype=torch.float32)[None, None]
        out = F.grid_sample(t, self.maps[k], mode="bilinear", align_corners=True)
        return out[0, 0].numpy()


def estimate_row_offset(c0: Camera, c1: Camera, pairs, search=np.arange(-3, 3.01, 0.25), max_disp=64, device=None) -> float:
    """The vertical misalignment of a calibrated rig, from images alone: the
    row shift of the right image that makes the census matching cost (at
    the best disparity, averaged over the image) smallest."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    costs = []
    for dv in search:
        r = Rectifier(c0, c1, row_offset=float(dv))
        total = 0.0
        for left, right in pairs:
            L = torch.tensor(r.rectify(left, 0), device=device)
            R = torch.tensor(r.rectify(right, 1), device=device)
            cl, cr = census(L), census(R)
            W = L.shape[1]
            best = None
            for d in range(max_disp):
                c = popcount(cl[:, max_disp:] ^ cr[:, max_disp - d : W - d])
                best = c if best is None else torch.minimum(best, c)
            total += float(best.float().mean())
        costs.append(total / len(pairs))
    return float(search[int(np.argmin(costs))])


def census(img: torch.Tensor, wh=(7, 5)) -> torch.Tensor:
    """Census transform: one bit per neighbor brighter than the center (int64)."""
    w, h = wh
    pad = F.pad(img[None, None], (w // 2, w // 2, h // 2, h // 2), mode="replicate")[0, 0]
    H, W = img.shape
    code = torch.zeros(H, W, dtype=torch.int64, device=img.device)
    bit = 0
    for dy in range(h):
        for dx in range(w):
            if dy == h // 2 and dx == w // 2:
                continue
            code |= (pad[dy : dy + H, dx : dx + W] > img).long() << bit
            bit += 1
    return code


def popcount(x: torch.Tensor) -> torch.Tensor:
    c = torch.zeros_like(x)
    for _ in range(35):
        c += x & 1
        x = x >> 1
    return c


def _aggregate(cost: torch.Tensor, dim: int, reverse: bool, P1: float, P2: float) -> torch.Tensor:
    """One SGM direction along dimension `dim` (1: rows -> along x, 2: along y) of cost [D, H, W]."""
    D = cost.shape[0]
    L = torch.empty_like(cost)
    n = cost.shape[dim]
    idx = range(n - 1, -1, -1) if reverse else range(n)
    prev = None
    for i in idx:
        c = cost.select(dim, i)  # [D, other]
        if prev is None:
            cur = c
        else:
            mn = prev.min(0, keepdim=True).values
            up = torch.cat([prev[1:], torch.full_like(prev[:1], float("inf"))])
            down = torch.cat([torch.full_like(prev[:1], float("inf")), prev[:-1]])
            cur = c + torch.minimum(torch.minimum(prev, torch.minimum(up, down) + P1), mn + P2) - mn
        L.select(dim, i).copy_(cur)
        prev = cur
    return L


def sgm_disparity(left: np.ndarray, right: np.ndarray, max_disp=64, P1=8.0, P2=96.0, uniqueness=0.95, device=None) -> np.ndarray:
    """Disparity of the left image (pixels; NaN where unreliable)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    L = torch.tensor(left, dtype=torch.float32, device=device)
    R = torch.tensor(right, dtype=torch.float32, device=device)
    cl, cr = census(L), census(R)
    H, W = L.shape
    cost = torch.full((max_disp, H, W), 35.0, device=device)
    for d in range(max_disp):
        cost[d, :, d:] = popcount(cl[:, d:] ^ cr[:, : W - d]).float()
    S = sum(_aggregate(cost, dim, rev, P1, P2) for dim in (2, 1) for rev in (False, True))
    best = S.argmin(0)
    vals = S.gather(0, best[None])[0]
    # sub-pixel: a parabola through the neighbors
    b = best.clamp(1, max_disp - 2)
    c0 = S.gather(0, (b - 1)[None])[0]
    c1 = S.gather(0, b[None])[0]
    c2 = S.gather(0, (b + 1)[None])[0]
    den = c0 - 2 * c1 + c2
    sub = torch.where(den > 0, 0.5 * (c0 - c2) / den, torch.zeros_like(den))
    disp = b.float() + sub
    # uniqueness: the second best (away from the best) must be clearly worse
    S2 = S.clone()
    for off in (-1, 0, 1):
        S2.scatter_(0, (best + off).clamp(0, max_disp - 1)[None], float("inf"))
    second = S2.min(0).values
    ok = (vals < uniqueness * second) & (best > 0) & (best < max_disp - 1)
    ok[:, :max_disp] = False
    disp[~ok] = float("nan")
    return disp.cpu().numpy()
