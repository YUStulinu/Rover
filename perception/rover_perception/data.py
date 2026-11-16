"""Data for perception: the simulated datasets (images with true depth and
surface labels) and EuRoC (images, calibration, ground-truth poses and the
Leica laser point cloud of the Vicon rooms, from which true depth and floor
labels are projected into the cameras)."""

from __future__ import annotations

import csv
import os
import re

import numpy as np
from PIL import Image

FLOOR, WALL, CEILING, OBSTACLE = 0, 1, 2, 3
CLASSES = ["floor", "wall", "ceiling", "obstacle"]


def read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def read_png16(path: str) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def yaml_numbers(text: str, key: str) -> list[float]:
    m = re.search(re.escape(key) + r":\s*\[([^\]]*)\]", text, re.S)
    if not m:
        m2 = re.search(re.escape(key) + r":\s*([-0-9.eE+]+)", text)
        return [float(m2.group(1))] if m2 else []
    return [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


class Camera:
    def __init__(self, yaml_path: str):
        y = open(yaml_path).read()
        fx, fy, cx, cy = yaml_numbers(y, "intrinsics")
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.dist = np.array(yaml_numbers(y, "distortion_coefficients") or [0, 0, 0, 0], dtype=float)
        self.size = tuple(int(v) for v in yaml_numbers(y, "resolution"))  # (w, h)
        T = np.array(yaml_numbers(y[y.find("T_BS"):], "data")).reshape(4, 4)
        self.T_bc = T

    def distort(self, x: np.ndarray, y: np.ndarray):
        k1, k2, p1, p2 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def quat_to_R(w, x, y, z):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class Sequence:
    """An EuRoC-format sequence (real or simulated)."""

    def __init__(self, root: str):
        self.root = root
        mav = os.path.join(root, "mav0")
        self.cam = [Camera(os.path.join(mav, f"cam{k}", "sensor.yaml")) for k in (0, 1)]
        self.frames = []  # (t_ns, left path, right path)
        right = {}
        with open(os.path.join(mav, "cam1", "data.csv")) as f:
            for r in csv.reader(f):
                if r and not r[0].startswith("#"):
                    right[r[0]] = r[1].strip()
        with open(os.path.join(mav, "cam0", "data.csv")) as f:
            for r in csv.reader(f):
                if r and not r[0].startswith("#") and r[0] in right:
                    self.frames.append((int(r[0]), os.path.join(mav, "cam0", "data", r[1].strip()), os.path.join(mav, "cam1", "data", right[r[0]])))
        gt = []
        with open(os.path.join(mav, "state_groundtruth_estimate0", "data.csv")) as f:
            for r in csv.reader(f):
                if r and not r[0].startswith("#"):
                    gt.append([float(v) for v in r[:8]])
        self.gt = np.array(gt)
        self.depth_dir = os.path.join(mav, "depth0", "data")
        self.label_dir = os.path.join(mav, "label0", "data")
        self.cloud_path = os.path.join(mav, "pointcloud0", "data.ply")

    def has_depth(self):
        return os.path.isdir(self.depth_dir)

    def truth(self, t_ns: int):
        """T_wb at a time (nearest ground-truth sample within 10 ms), or None."""
        i = np.searchsorted(self.gt[:, 0], t_ns)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.gt) and abs(self.gt[j, 0] - t_ns) < 1e7:
                if best is None or abs(self.gt[j, 0] - t_ns) < abs(self.gt[best, 0] - t_ns):
                    best = j
        if best is None:
            return None
        g = self.gt[best]
        T = np.eye(4)
        T[:3, :3] = quat_to_R(*g[4:8])
        T[:3, 3] = g[1:4]
        return T

    def cloud(self) -> np.ndarray:
        """The laser point cloud (N x 3, world frame), cached as .npy next to the PLY."""
        cache = self.cloud_path + ".npy"
        if os.path.exists(cache):
            return np.load(cache)
        with open(self.cloud_path, "rb") as f:
            n = 0
            while True:
                line = f.readline().decode()
                if line.startswith("element vertex"):
                    n = int(line.split()[2])
                if line.startswith("end_header"):
                    break
            data = np.loadtxt(f, usecols=(0, 1, 2), max_rows=n, dtype=np.float32)
        np.save(cache, data)
        return data
