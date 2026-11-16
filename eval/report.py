"""Collect every measured result into results/README.md: the odometry and SLAM
on EuRoC against published systems, the self-calibration ablation, the
odometry on simulated worlds, the missions and the perception.

    python eval/report.py   (from the repository root)
"""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cm(v):
    return "—" if v is None else f"{100 * v:.1f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def euroc_section(out):
    pub = load(os.path.join(ROOT, "data", "published_euroc.json"))
    seqs = pub["sequences"]
    ours = {s: load(os.path.join(R, "vio", s + ".json")) for s in seqs}
    short = [s.split("_")[0] + s.split("_")[1] for s in seqs]
    out.append("## EuRoC MAV: absolute trajectory error (RMS, cm)\n")
    out.append("SE(3) alignment of the whole trajectory to the ground truth (no scale), every frame. "
               "Published numbers: ORB-SLAM3 paper, Table II (see `data/published_euroc.json`); a dash: not completed or not reported.\n")
    out.append("| system | " + " | ".join(short) + " | mean |")
    out.append("|---" * (len(seqs) + 2) + "|")

    def row(name, vals, bold=False):
        m = mean(vals) if all(v is not None for v in vals) else None
        cells = [cm(v) for v in vals] + [cm(m)]
        if bold:
            name = f"**{name}**"
            cells = [f"**{c}**" for c in cells]
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    row("Rover, stereo-inertial + loops (final)", [ours[s]["ate_final_m"] if ours[s] else None for s in seqs], True)
    row("Rover, stereo-inertial odometry only", [ours[s]["ate_odometry_m"] if ours[s] else None for s in seqs])
    for group, title in (("stereo_inertial", "stereo-inertial"), ("monocular_inertial", "mono-inertial"), ("stereo", "stereo")):
        for name, vals in pub[group].items():
            row(f"{name} ({title})", vals)
    out.append("")
    out.append("| sequence | length (m) | keyframes | loops (correct) | RPE (%) | RPE (°/m) | front-end (ms) | back-end (ms) | stereo row offset (px) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for s in seqs:
        r = ours[s]
        if not r:
            continue
        out.append(f"| {s} | {r['length_m']:.1f} | {r['keyframes']} | {r['loops']} ({r['loops_correct']}) | {r['rpe_trans_pct']:.2f} | "
                   f"{r['rpe_rot_deg_per_m']:.3f} | {r['frontend_ms']:.1f} | {r['backend_ms']:.1f} | {r.get('stereo_row_offset_px', 0):+.2f} |")
    out.append("\nA loop is counted correct when its relative pose is within 30 cm and 5° of the ground truth.\n")
    ab = {s: load(os.path.join(R, "ablation_noselfcal", s + ".json")) for s in seqs}
    if any(ab.values()):
        out.append("### Ablation: stereo self-calibration off\n")
        out.append("| sequence | ATE final, off (cm) | ATE final, on (cm) | loops off | loops on |")
        out.append("|---|---|---|---|---|")
        for s in seqs:
            a, b = ab[s], ours[s]
            if a and b:
                out.append(f"| {s} | {cm(a['ate_final_m'])} | {cm(b['ate_final_m'])} | {a['loops']} | {b['loops']} |")
        out.append("")


def sim_section(out):
    files = sorted(glob.glob(os.path.join(R, "sim_vio", "world*.json")))
    if files:
        out.append("## Odometry on simulated worlds\n")
        out.append("| world | frames | length (m) | ATE odometry (cm) | ATE final (cm) | loops (correct) | keyframes |")
        out.append("|---|---|---|---|---|---|---|")
        for f in files:
            r = load(f)
            out.append(f"| {os.path.basename(f)[:-5]} | {r['frames']} | {r['length_m']:.1f} | {cm(r['ate_odometry_m'])} | {cm(r['ate_final_m'])} | "
                       f"{r['loops']} ({r['loops_correct']}) | {r['keyframes']} |")
        out.append("")
    runs = sorted(glob.glob(os.path.join(R, "missions", "seed*_*.json")))
    if runs:
        out.append("## Missions: planning and MPC in the simulator\n")
        out.append("Seven goals per world (three random, three corners, the start); A* on an occupancy grid inflated by the rover's radius plus 25 cm, "
                   "then the MPC. The pose is either the simulator's truth or the stereo-inertial odometry running in the loop.\n")
        out.append("| world | localization | goals | collision | time (s) | travelled (m) | cross-track RMS (cm) | min clearance (cm) | localization RMS (cm) | MPC mean / max (ms) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for f in runs:
            m = load(f)["metrics"]
            out.append(f"| {m['seed']} | {m['localization']} | {m['reached']}/{m['goals']} | {'yes' if m['collided'] else 'no'} | {m['seconds']:.1f} | "
                       f"{m['travelled_m']:.1f} | {cm(m['cross_track_rms_m'])} | {cm(m['min_clearance_m'])} | {cm(m['localization_rms_m'])} | "
                       f"{m['mpc_ms_mean']:.1f} / {m['mpc_ms_max']:.1f} |")
        out.append("")


def perception_section(out):
    p = load(os.path.join(R, "perception", "perception.json"))
    if not p:
        return
    out.append("## Perception\n")
    tr = p.get("training")
    if tr:
        h = tr["history"][-1]
        out.append(f"U-Net trained on {tr['images']} simulated images ({', '.join(tr['worlds'])}), {h['epoch']} epochs, {h['seconds'] / 60:.0f} min.\n")
    st = p.get("sim_test")
    if st:
        out.append("### Unseen simulated world (world5): depth up to 10 m\n")
        out.append("| method | AbsRel | RMSE (m) | δ<1.25 |")
        out.append("|---|---|---|---|")
        names = {"mono": "U-Net, metric", "mono_median": "U-Net, median-scaled", "sgm": "SGM stereo (metric)", "dav2": "Depth Anything V2 Small, scale+shift fitted to truth"}
        for k, n in names.items():
            v = st.get(k)
            if v:
                out.append(f"| {n} | {v['abs_rel']:.3f} | {v['rmse']:.3f} | {v['delta1']:.3f} |")
        out.append("\n| class | IoU |\n|---|---|")
        for c, v in st["iou"].items():
            out.append(f"| {c} | {v:.3f} |")
        out.append(f"| **mean** | **{st['miou']:.3f}** |\n")
    eu = p.get("euroc")
    if eu:
        out.append("### EuRoC Vicon rooms: depth against the Leica laser scan (AbsRel / δ<1.25)\n")
        keys = [("mono", "U-Net metric"), ("mono_median", "U-Net median-scaled"), ("mono_aligned", "U-Net scale+shift"),
                ("sgm", "SGM"), ("sgm_selfcal", "SGM, row offset self-calibrated"), ("dav2", "Depth Anything V2 scale+shift")]
        keys = [(k, n) for k, n in keys if any(eu[s].get(k) for s in eu)]
        out.append("| sequence | " + " | ".join(n for _, n in keys) + " | floor IoU (U-Net) |")
        out.append("|---" * (len(keys) + 2) + "|")
        for s, r in eu.items():
            cells = [f"{r[k]['abs_rel']:.3f} / {r[k]['delta1']:.2f}" if r.get(k) else "—" for k, _ in keys]
            out.append(f"| {s} | " + " | ".join(cells) + f" | {r['floor_iou']:.2f} |")
        out.append("")


def main():
    out = ["# Rover: measured results\n", "Generated by `eval/report.py` from the JSON files in this directory.\n"]
    euroc_section(out)
    sim_section(out)
    perception_section(out)
    with open(os.path.join(R, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote results/README.md ({len(out)} lines)")


if __name__ == "__main__":
    main()
