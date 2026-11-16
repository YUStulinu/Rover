"""Train the U-Net on simulated worlds and evaluate it (with SGM stereo and
Depth Anything V2 Small as baselines) on an unseen simulated world and on
EuRoC Vicon-room sequences against the laser point cloud.

    python run_perception.py --train data/sim/world2 data/sim/world3 data/sim/world4 \
        --test data/sim/world5 --euroc data/euroc/V1_01_easy ... --out results/perception
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rover_perception import experiments as ex  # noqa: E402
from rover_perception.net import UNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--euroc", nargs="*", default=[])
    ap.add_argument("--out", default="results/perception")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--weights", default=None, help="skip training, load these weights")
    ap.add_argument("--no-dav2", action="store_true")
    ap.add_argument("--euroc-every", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}
    out_json = os.path.join(a.out, "perception.json")
    if a.weights and os.path.exists(out_json):  # re-evaluation: keep the training record
        with open(out_json) as f:
            report = json.load(f)
    if a.weights:
        net = UNet()
        net.load_state_dict(torch.load(a.weights, map_location="cpu"))
        net = net.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        path = os.path.join(a.out, "unet.pt")
        net, hist, n = ex.train(a.train, path, epochs=a.epochs, log=lambda m: print(m, flush=True))
        report["training"] = {"worlds": [os.path.basename(r) for r in a.train], "images": n, "history": hist}
    net.eval()
    dav2 = None if a.no_dav2 else ex.DepthAnything()
    if a.test:
        report["sim_test"] = ex.evaluate_sim(net, a.test, dav2)
        print("sim test", json.dumps(report["sim_test"]), flush=True)
    if a.euroc:
        report["euroc"] = ex.evaluate_euroc(net, a.euroc, dav2, every=a.euroc_every, log=lambda *m: print(*m, flush=True))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
