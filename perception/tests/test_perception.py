import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rover_perception import experiments as ex  # noqa: E402
from rover_perception.net import UNet, silog_loss  # noqa: E402
from rover_perception.stereo import census, popcount, sgm_disparity  # noqa: E402


def test_popcount():
    x = torch.tensor([0, 1, 3, 255, (1 << 34) - 1])
    assert popcount(x).tolist() == [0, 1, 2, 8, 34]


def test_census_constant_image_is_zero():
    assert census(torch.full((10, 12), 7.0)).abs().sum() == 0


def test_sgm_recovers_shift():
    rng = np.random.default_rng(0)
    H, W, d = 60, 160, 9
    # a smooth random texture
    base = rng.random((H, W + d)) * 255
    k = np.ones(3) / 3
    base = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, base)
    left = base[:, :W]
    right = base[:, d : d + W]  # left x appears at x - d on the right
    disp = sgm_disparity(left, right, max_disp=32, device="cpu")
    core = disp[5:-5, 40:-5]
    ok = np.isfinite(core)
    assert ok.mean() > 0.8
    assert np.median(np.abs(core[ok] - d)) < 0.3


def test_unet_shapes_and_range():
    net = UNet(base=8).eval()
    log_d, seg = net(torch.zeros(2, 1, 50, 70))
    assert log_d.shape == (2, 1, 50, 70) and seg.shape == (2, 4, 50, 70)
    d = torch.exp(log_d)
    assert d.min() >= UNet.MIN_DEPTH - 1e-5 and d.max() <= UNet.MAX_DEPTH + 1e-4


def test_silog_is_zero_for_exact_and_scale_tolerant():
    depth = torch.rand(1, 1, 8, 8) * 5 + 1
    valid = torch.ones_like(depth, dtype=torch.bool)
    assert silog_loss(torch.log(depth), depth, valid) < 1e-3
    exact = silog_loss(torch.log(depth), depth, valid)
    scaled = silog_loss(torch.log(depth * 2), depth, valid)
    assert scaled > exact


def test_depth_metrics_and_alignment():
    gt = np.linspace(1, 5, 100).reshape(10, 10)
    mask = np.ones_like(gt, bool)
    m = ex.depth_metrics(gt, gt, mask)
    assert m["abs_rel"] == 0 and m["delta1"] == 1
    # an affine map of inverse depth is undone by the alignment
    rel = 3.0 / gt + 0.2
    back = ex.align_inverse(rel, gt, mask)
    assert np.allclose(back, gt, rtol=1e-6)
    assert np.allclose(ex.median_scaled(gt * 2, gt, mask), gt)


def test_iou():
    a = np.array([[0, 0], [1, 1]])
    b = np.array([[0, 1], [1, 1]])
    assert ex.iou(a, b, 0) == 0.5
    assert abs(ex.iou(a, b, 1) - 2 / 3) < 1e-12
    assert np.isnan(ex.iou(a, b, 3))


def _camera(x):
    from rover_perception.data import Camera

    c = Camera.__new__(Camera)
    c.K = np.array([[230.0, 0, 188], [0, 230.0, 120], [0, 0, 1]])
    c.dist = np.zeros(4)
    c.size = (376, 240)
    c.T_bc = np.eye(4)
    c.T_bc[0, 3] = x
    return c


def test_row_offset_of_a_misaligned_rig():
    from rover_perception.stereo import estimate_row_offset

    rng = np.random.default_rng(1)
    big = rng.random((260, 420)) * 255
    k = np.ones(5) / 5
    big = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, big)
    big = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, big)
    left = big[10:250, 20:396].astype(np.float32)
    right = big[8:248, 28:404].astype(np.float32)  # 8 px disparity, content 2 rows lower
    c0, c1 = _camera(0.0), _camera(0.11)
    dv = estimate_row_offset(c0, c1, [(left, right)], search=np.arange(-3, 3.01, 0.5), max_disp=16, device="cpu")
    assert dv == 2.0
