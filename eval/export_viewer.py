"""Export the results for the 3D viewer (ui/): one JSON scene per EuRoC
sequence (the ground truth, the estimated trajectory aligned to it, the
closed loops and, in the Vicon rooms, the laser point cloud) and one per
simulated mission (the world, the planned paths, the true and the estimated
trajectories).

    python eval/export_viewer.py   (from the repository root)
"""

import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ui", "data")


def r3(a):
    return [round(float(v), 3) for v in np.asarray(a).ravel()]


def align_rigid(est, gt):
    """R, t minimizing |R est + t - gt| (Horn / Kabsch)."""
    me, mg = est.mean(0), gt.mean(0)
    H = (est - me).T @ (gt - mg)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mg - R @ me


def read_truth(seq):
    rows = []
    with open(os.path.join(ROOT, "data", "euroc", seq, "mav0", "state_groundtruth_estimate0", "data.csv")) as f:
        for r in csv.reader(f):
            if r and not r[0].startswith("#"):
                rows.append([float(r[0]) * 1e-9, float(r[1]), float(r[2]), float(r[3])])
    return np.array(rows)


def cloud(seq, n=40000, seed=0):
    ply = os.path.join(ROOT, "data", "euroc", seq, "mav0", "pointcloud0", "data.ply")
    cache = ply + ".npy"
    if os.path.exists(cache):
        pts = np.load(cache)
    elif os.path.exists(ply):
        with open(ply, "rb") as f:
            while not f.readline().startswith(b"end_header"):
                pass
            pts = np.loadtxt(f, usecols=(0, 1, 2), dtype=np.float32)
    else:
        return None
    idx = np.random.default_rng(seed).choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[np.sort(idx)]


def euroc_scene(seq):
    est = np.loadtxt(os.path.join(ROOT, "results", "traj", seq + ".txt"))
    gt = read_truth(seq)
    # ground truth at the estimate's times (nearest sample within 5 ms)
    j = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 1, len(gt) - 1)
    j = np.where(np.abs(gt[j - 1, 0] - est[:, 0]) < np.abs(gt[j, 0] - est[:, 0]), j - 1, j)
    ok = np.abs(gt[j, 0] - est[:, 0]) < 0.005
    R, t = align_rigid(est[ok, 1:4], gt[j[ok], 1:4])
    p = est[:, 1:4] @ R.T + t
    err = np.linalg.norm(p - gt[j, 1:4], axis=1)
    err[~ok] = np.nan
    every = 2  # 20 Hz -> 10 Hz
    k = np.arange(0, len(est), every)
    g = gt[:: max(1, len(gt) // (len(k) or 1))]
    loops = []
    lp = os.path.join(ROOT, "results", "loops", seq + ".txt")
    if os.path.exists(lp) and os.path.getsize(lp) > 0:
        for ti, tj, good in np.atleast_2d(np.loadtxt(lp)):
            a = p[np.argmin(np.abs(est[:, 0] - ti))]
            b = p[np.argmin(np.abs(est[:, 0] - tj))]
            loops.append({"a": r3(a), "b": r3(b), "good": bool(good), "t": round(float(tj - est[0, 0]), 2)})
    metrics = json.load(open(os.path.join(ROOT, "results", "vio", seq + ".json")))
    metrics.pop("sequence", None)
    scene = {
        "name": seq,
        "kind": "euroc",
        "metrics": metrics,
        "tracks": [
            {"name": "ground truth (Vicon / Leica)", "role": "truth", "t": r3(g[:, 0] - est[0, 0]), "xyz": r3(g[:, 1:4])},
            {"name": "estimate (aligned, SE(3))", "role": "estimate", "t": r3(est[k, 0] - est[0, 0]), "xyz": r3(p[k]), "error": [None if np.isnan(e) else round(float(e), 3) for e in err[k]]},
        ],
        "loops": loops,
    }
    c = cloud(seq) if seq.startswith("V") else None
    if c is not None:
        scene["cloud"] = r3(c)
    return scene


def mission_scene(seed):
    runs = {}
    for tag in ("gt", "vio"):
        path = os.path.join(ROOT, "results", "missions", f"seed{seed}_{tag}.json")
        if os.path.exists(path):
            runs[tag] = json.load(open(path))
    if "vio" not in runs:
        return None
    v = runs["vio"]
    z = 0.0
    tr = v["trajectory"]
    t0 = tr["t"][0]
    tracks = [
        {"name": "true path, driven on the odometry", "role": "truth", "t": [round(t - t0, 3) for t in tr["t"]], "xyz": [c for x, y in zip(tr["x"], tr["y"]) for c in (x, y, z)]},
        {
            "name": "odometry estimate",
            "role": "estimate",
            "t": [round(t - t0, 3) for t in tr["t"]],
            "xyz": [c for x, y in zip(tr["ex"], tr["ey"]) for c in (x, y, z)],
            "error": [round(float(np.hypot(a - c, b - d)), 3) for a, b, c, d in zip(tr["ex"], tr["ey"], tr["x"], tr["y"])],
        },
    ]
    if "gt" in runs:
        g = runs["gt"]["trajectory"]
        tracks.append({"name": "path driven on the true pose", "role": "reference", "t": [round(t - g["t"][0], 3) for t in g["t"]], "xyz": [c for x, y in zip(g["x"], g["y"]) for c in (x, y, z)]})
    return {
        "name": f"mission, world seed {seed}",
        "kind": "mission",
        "metrics": {"odometry": v["metrics"], "ground truth": runs.get("gt", {}).get("metrics")},
        "tracks": tracks,
        "paths": [[c for p in path for c in (p[0], p[1], 0.01)] for path in v["paths"]],
        "room": v["world"]["size"],
        "boxes": v["world"]["boxes"],
        "loops": [],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "traj", "*.txt"))):
        seq = os.path.basename(f)[:-4]
        s = euroc_scene(seq)
        json.dump(s, open(os.path.join(OUT, seq + ".json"), "w"), separators=(",", ":"))
        index.append({"file": seq + ".json", "name": seq, "kind": "euroc"})
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "missions", "seed*_vio.json"))):
        seed = int(os.path.basename(f)[4:].split("_")[0])
        s = mission_scene(seed)
        if s:
            name = f"mission_seed{seed}.json"
            json.dump(s, open(os.path.join(OUT, name), "w"), separators=(",", ":"))
            index.append({"file": name, "name": s["name"], "kind": "mission"})
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(f"{len(index)} scenes in {OUT}")


if __name__ == "__main__":
    main()
