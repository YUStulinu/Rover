"""Training the network on simulated worlds, and the evaluation of depth and
segmentation: on a simulated world never seen in training, and on real
EuRoC images against the Leica laser point cloud."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import FLOOR, Sequence, read_gray, read_png16
from .net import UNet, silog_loss
from .stereo import Rectifier, estimate_row_offset, sgm_disparity


# ---- simulated data ------------------------------------------------------------


def sim_samples(roots, every=1):
    out = []
    for root in roots:
        s = Sequence(root)
        for k, (t, left, right) in enumerate(s.frames):
            if k % every:
                continue
            name = os.path.basename(left)
            out.append((left, right, os.path.join(s.depth_dir, name), os.path.join(s.label_dir, name)))
    return out


def load_sim(sample):
    left, right, dpath, lpath = sample
    img = read_gray(left).astype(np.float32) / 255
    depth = read_png16(dpath).astype(np.float32) / 1000
    label = read_gray(lpath).astype(np.int64)
    return img, depth, label, read_gray(right).astype(np.float32) / 255


def train(train_roots, out_path, epochs=15, batch=12, lr=2e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = sim_samples(train_roots)
    # compact storage: 8-bit images and labels, depth in millimeters
    imgs = torch.tensor(np.stack([read_gray(s[0]) for s in samples]))[:, None]
    depths = torch.tensor(np.stack([read_png16(s[2]).astype(np.int32) for s in samples]))[:, None]
    labels = torch.tensor(np.stack([read_gray(s[3]) for s in samples]))
    data = samples
    net = UNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(data) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    history = []
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(len(data))
        tot = 0.0
        for b in range(len(data) // batch):
            idx = perm[b * batch : (b + 1) * batch]
            x = imgs[idx].to(device).float() / 255
            d = depths[idx].to(device).float() / 1000
            y = labels[idx].to(device).long()
            # photometric augmentation: gain, offset, gamma, noise (the real camera differs)
            g = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.6, 1.4)
            o = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(-0.15, 0.15)
            gam = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.7, 1.4)
            x = ((x.clamp(1e-4, 1) ** gam) * g + o + 0.02 * torch.randn_like(x)).clamp(0, 1)
            # horizontal flips
            if rng.random() < 0.5:
                x, d, y = x.flip(-1), d.flip(-1), y.flip(-1)
            log_d, seg = net((x - 0.5) / 0.25)
            valid = (d > 0.1) & (d < 20)
            loss = silog_loss(log_d, d, valid) + 0.5 * F.cross_entropy(seg, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += loss.item()
        history.append({"epoch": ep + 1, "loss": tot / max(1, len(data) // batch), "seconds": time.time() - t0})
        log(history[-1])
    torch.save(net.state_dict(), out_path)
    return net, history, len(data)


# ---- metrics ----------------------------------------------------------------------


def depth_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(g) == 0:
        return None
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "points": int(len(g)),
    }


def align_inverse(rel_inv, gt, mask):
    """Scale and shift of a relative inverse-depth map fitted to the true
    inverse depth by least squares (the usual protocol for affine-invariant
    depth models; it uses the ground truth)."""
    a = rel_inv[mask]
    b = 1.0 / gt[mask]
    A = np.stack([a, np.ones_like(a)], 1)
    s, t = np.linalg.lstsq(A, b, rcond=None)[0]
    inv = s * rel_inv + t
    return 1.0 / np.clip(inv, 1 / 20, 1 / 0.1)


def median_scaled(pred, gt, mask):
    return pred * np.median(gt[mask]) / np.median(pred[mask])


def mean_of(ms):
    ms = [m for m in ms if m]
    if not ms:
        return None
    keys = ["abs_rel", "rmse", "delta1"]
    out = {k: float(np.average([m[k] for m in ms], weights=[m["points"] for m in ms])) for k in keys}
    out["images"] = len(ms)
    return out


def iou(pred, gt, cls, mask=None):
    m = np.ones_like(gt, bool) if mask is None else mask
    a, b = (pred == cls) & m, (gt == cls) & m
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


# ---- the depth-anything baseline --------------------------------------------------


class DepthAnything:
    """Depth Anything V2 Small (Hugging Face weights), zero shot: relative inverse depth."""

    def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.net = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, gray01: np.ndarray) -> np.ndarray:
        rgb = np.repeat((gray01 * 255).astype(np.uint8)[..., None], 3, -1)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        out = self.net(**inp).predicted_depth  # relative inverse depth
        out = F.interpolate(out[:, None], size=gray01.shape, mode="bilinear", align_corners=False)[0, 0]
        return out.cpu().numpy()


# ---- evaluation ---------------------------------------------------------------------


@torch.no_grad()
def predict(net, img01):
    device = next(net.parameters()).device
    x = torch.tensor(img01, dtype=torch.float32, device=device)[None, None]
    log_d, seg = net((x - 0.5) / 0.25)
    return torch.exp(log_d)[0, 0].cpu().numpy(), seg.argmax(1)[0].cpu().numpy()


def evaluate_sim(net, roots, dav2=None, every=5, max_depth=10.0):
    samples = sim_samples(roots, every)
    rect = None
    res = {"mono": [], "mono_median": [], "sgm": [], "dav2": []}
    ious = {c: [] for c in range(4)}
    seq = Sequence(roots[0])
    rect = Rectifier(seq.cam[0], seq.cam[1], size=seq.cam[0].size, f=seq.cam[0].K[0, 0])
    for s in samples:
        img, depth, label, right = load_sim(s)
        mask = (depth > 0.3) & (depth < max_depth)
        d, seg = predict(net, img)
        res["mono"].append(depth_metrics(d, depth, mask))
        res["mono_median"].append(depth_metrics(median_scaled(d, depth, mask), depth, mask))
        disp = sgm_disparity(img * 255, right * 255)
        sd = rect.K[0, 0] * rect.baseline / disp
        m2 = mask & np.isfinite(sd)
        res["sgm"].append(depth_metrics(sd, depth, m2))
        if dav2 is not None:
            res["dav2"].append(depth_metrics(align_inverse(dav2(img), depth, mask), depth, mask))
        for c in range(4):
            ious[c].append(iou(seg, label, c))
    out = {k: mean_of(v) for k, v in res.items()}
    out["sgm_density"] = None
    out["iou"] = {name: float(np.nanmean(ious[c])) for c, name in enumerate(["floor", "wall", "ceiling", "obstacle"])}
    out["miou"] = float(np.nanmean(list(out["iou"].values())))
    return out


def laser_views(seq: Sequence, rect: Rectifier, frame, cloud, floor_z):
    """True depth and floor mask in the rectified left camera, from the laser cloud."""
    t_ns = frame[0]
    T_wb = seq.truth(t_ns)
    if T_wb is None:
        return None
    T_wc = T_wb @ seq.cam[0].T_bc
    R_wc, p_wc = T_wc[:3, :3], T_wc[:3, 3]
    Xc = (cloud - p_wc) @ R_wc  # world -> left camera
    Xr = Xc @ rect.R0.T  # -> rectified
    z = Xr[:, 2]
    keep = (z > 0.2) & (z < 20)
    Xr, zk, wz = Xr[keep], z[keep], cloud[keep, 2]
    u = rect.K[0, 0] * Xr[:, 0] / zk + rect.K[0, 2]
    v = rect.K[1, 1] * Xr[:, 1] / zk + rect.K[1, 2]
    w, h = rect.size
    inside = (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
    u, v, zk, wz = u[inside], v[inside], zk[inside], wz[inside]
    depth = np.full((h, w), np.inf)
    floor = np.zeros((h, w), bool)
    order = np.argsort(-zk)  # far first, so that near points overwrite
    ui, vi = u[order].astype(int), v[order].astype(int)
    zz, ff = zk[order], wz[order] < floor_z + 0.08
    for du in (0, 1):  # 2 x 2 splats against gaps between points
        for dv in (0, 1):
            depth[vi + dv, ui + du] = zz
            floor[vi + dv, ui + du] = ff
    # a z-buffer by minimum would need a loop; the far-to-near order gives the same result
    depth[~np.isfinite(depth)] = 0
    return depth, floor


def evaluate_euroc(net, roots, dav2=None, every=40, max_depth=10.0, log=print):
    per_seq = {}
    for root in roots:
        seq = Sequence(root)
        rect = Rectifier(seq.cam[0], seq.cam[1])
        # the rig's vertical misalignment, from six image pairs (no ground truth)
        idx = np.linspace(200, len(seq.frames) - 200, 6).astype(int)
        pairs = [(read_gray(seq.frames[k][1]).astype(np.float32), read_gray(seq.frames[k][2]).astype(np.float32)) for k in idx]
        offset = estimate_row_offset(seq.cam[0], seq.cam[1], pairs)
        rect_sc = Rectifier(seq.cam[0], seq.cam[1], row_offset=offset)
        cloud = seq.cloud()
        floor_z = float(np.percentile(cloud[:, 2], 2))
        res = {"mono": [], "mono_median": [], "sgm": [], "sgm_selfcal": [], "dav2": [], "mono_aligned": []}
        floor_iou = []
        for k in range(0, len(seq.frames), every):
            f = seq.frames[k]
            views = laser_views(seq, rect, f, cloud, floor_z)
            if views is None:
                continue
            gt, floor = views
            L = rect.rectify(read_gray(f[1]).astype(np.float32), 0)
            R = rect.rectify(read_gray(f[2]).astype(np.float32), 1)
            img = L / 255.0
            mask = (gt > 0.3) & (gt < max_depth)
            if mask.sum() < 500:
                continue
            d, seg = predict(net, img)
            res["mono"].append(depth_metrics(d, gt, mask))
            res["mono_median"].append(depth_metrics(median_scaled(d, gt, mask), gt, mask))
            res["mono_aligned"].append(depth_metrics(align_inverse(1.0 / d, gt, mask), gt, mask))
            disp = sgm_disparity(L, R)
            sd = rect.K[0, 0] * rect.baseline / disp
            m2 = mask & np.isfinite(sd)
            res["sgm"].append(depth_metrics(sd, gt, m2))
            R2 = rect_sc.rectify(read_gray(f[2]).astype(np.float32), 1)
            sd2 = rect.K[0, 0] * rect.baseline / sgm_disparity(L, R2)
            res["sgm_selfcal"].append(depth_metrics(sd2, gt, mask & np.isfinite(sd2)))
            if dav2 is not None:
                res["dav2"].append(depth_metrics(align_inverse(dav2(img), gt, mask), gt, mask))
            floor_iou.append(iou((seg == FLOOR).astype(int), floor.astype(int), 1, mask))
        per_seq[os.path.basename(root)] = {k: mean_of(v) for k, v in res.items()} | {
            "floor_iou": float(np.nanmean(floor_iou)) if floor_iou else None,
            "row_offset_px_rectified": offset,
        }
        log(os.path.basename(root), per_seq[os.path.basename(root)])
    return per_seq
