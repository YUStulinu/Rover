# Rover

Autonomy for a small ground robot, built from scratch in five languages:
stereo visual-inertial odometry and graph SLAM with its own sparse solver
(C++), a simulator (Rust), planning and model predictive control (Julia),
depth and surface perception (Python) and a 3D viewer (TypeScript/WebGL).
The odometry is evaluated on the real EuRoC MAV dataset against published
systems; the whole loop (simulator → odometry → planner → MPC → simulator)
runs in simulated worlds.

All numbers below were measured on one machine (Windows 11, 8-core CPU,
RTX GPU for the network only) and are collected in
[`results/README.md`](results/README.md) by `eval/report.py`.

## What is in it

| part | language | what it does |
|---|---|---|
| `vio/` | C++20, no dependencies | Lie-group math, sparse LDLᵀ (elimination tree, exact minimum degree, symbolic/numeric split), Levenberg–Marquardt on manifolds, own PNG/zlib decoder, FAST + pyramidal KLT + ORB, IMU preintegration with bias Jacobians, a 10-keyframe sliding window with inverse-depth landmarks and Schur-complement marginalization, a bag-of-words vocabulary, 3D-3D RANSAC loop verification, a 4-DoF pose graph, ATE/RPE |
| `sim/` | Rust | a unicycle rover with actuator lag in a textured room with box obstacles; ray-cast stereo camera (2×2 supersampling), IMU with the ADIS16448 noise model, EuRoC-format writer with true depth and surface labels, a JSON step server for closed-loop control |
| `mpc/` | Julia, stdlib only | occupancy grid, A* with path shortcutting, an ADMM QP solver (OSQP-style), a sequential-QP unicycle MPC (20 steps × 0.1 s, rate limits, obstacle half-planes), the closed loop that drives the simulator with either the true pose or the odometry |
| `perception/` | Python, PyTorch | a U-Net (depth + 4-class segmentation) trained only in simulation; Fusiello rectification; census-cost semi-global matching on the GPU; laser-scan ground truth for EuRoC; Depth Anything V2 Small as a pre-trained baseline |
| `ui/` | TypeScript, WebGL2, no 3D library | orbit viewer of trajectories (colored by error), loop closures, the Leica scans and the missions, with time playback |

## Results

### EuRoC MAV: absolute trajectory error (RMS, cm)

Stereo-inertial, every frame of all 11 sequences, SE(3)-aligned; the other
rows are from the ORB-SLAM3 paper (Table II).

| system | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Rover, with loop closure** | **3.2** | **4.5** | **4.8** | **8.9** | **11.2** | **3.7** | **3.6** | **4.5** | **5.3** | **8.8** | **8.5** | **6.1** |
| Rover, odometry only | 7.1 | 7.0 | 7.8 | 10.6 | 15.5 | 6.5 | 5.7 | 4.5 | 7.9 | 8.9 | 19.9 | 9.2 |
| VINS-Fusion (stereo-inertial) | 16.6 | 15.2 | 12.5 | 28.0 | 28.4 | 7.6 | 6.9 | 11.4 | 6.6 | 9.1 | 9.6 | 13.8 |
| Kimera | 8.0 | 9.0 | 11.0 | 15.0 | 24.0 | 5.0 | 11.0 | 12.0 | 7.0 | 10.0 | 19.0 | 11.9 |
| BASALT | 8.0 | 6.0 | 5.0 | 10.0 | 8.0 | 4.0 | 2.0 | 3.0 | 3.0 | 2.0 | — | — |
| ORB-SLAM3 (stereo-inertial) | 3.6 | 3.3 | 3.5 | 5.1 | 8.2 | 3.8 | 1.4 | 2.4 | 3.2 | 1.4 | 2.4 | 3.5 |

Rover is better than VINS-Fusion and Kimera on average, but clearly behind
ORB-SLAM3 and BASALT, above all in the Vicon rooms at medium and high speed.
307 loop closures were accepted, and 305 of them are within 30 cm and 5° of
the ground truth (the two wrong ones are in MH02). Per stereo frame, on one
core: front-end 36–55 ms, back-end 20–42 ms (four sequences ran in parallel).

**A calibration finding.** In the V2 room no loop was ever verified. The
cause: with the official calibration the right image of every V2 sequence is
about 1.5 px off its epipolar line (V1 and Machine Hall: under 0.4 px), so
most stereo matches failed the 1.5 px epipolar gate. The front-end now
estimates this row offset online from its own KLT matches (a small rotation
of the rig about the baseline). With it, V2 goes from 0 to 37 loops (all
correct) and its final error falls from 6.9/12.7/16.4 cm to 5.3/8.8/8.5 cm.
The Python stereo code measured the offset independently, from the images
alone. Its correction improves SGM depth in V2 (AbsRel 0.22/0.22/0.30 →
0.14/0.17/0.22) and changes nothing in V1. The odometry without loops gets
worse on two V2 sequences, which is reported as it is (see the ablation in
`results/README.md`).

### Simulation

| | |
|---|---|
| odometry in 4 unseen worlds, 30 m each | ATE 2.9–3.8 cm (one lap, so no loop is closed) |
| missions in 5 worlds (7 goals, 42–47 m each), true pose | 35/35 goals, no collision, true cross-track RMS 4.4–4.7 cm, min clearance ≥ 24 cm |
| the same missions, the odometry in the loop | 35/35 goals, no collision, true cross-track RMS 5.9–7.5 cm, localization RMS 3.9–8.3 cm, min clearance ≥ 19.5 cm |
| MPC solve time | 4.4–7.0 ms mean, 110 ms worst |

The cross-track error is always measured with the true position, also when the controller only sees the odometry.

### Perception

The U-Net is trained only on simulated images (3 worlds, 3378 images, 20 epochs).

| | depth AbsRel | δ<1.25 | segmentation |
|---|---|---|---|
| unseen simulated world, U-Net (metric) | 0.019 | 0.998 | mIoU 0.88 |
| unseen simulated world, SGM stereo | 0.034 | 0.998 | — |
| EuRoC V1/V2 vs laser, U-Net (metric) | 0.48–0.58 | 0.07–0.22 | floor IoU 0.32–0.49 |
| EuRoC V1/V2 vs laser, SGM (self-calibrated) | 0.13–0.22 | 0.74–0.86 | — |
| EuRoC V1/V2 vs laser, Depth Anything V2 Small (scale and shift fitted to the truth) | 0.07–0.13 | 0.84–0.94 | — |

The network trained in simulation does not transfer to the real rooms. The
simulated textures and lighting are too unlike EuRoC. Metric depth is wrong
by about 50%, and even median-scaled it stays at AbsRel 0.27–0.45. Classical
stereo needs no training and does much better. The large pre-trained model
is the best at relative depth. It gets an advantage here, because its scale
and shift are fitted to the ground truth in each image.

## Running it

Requirements: a C++20 compiler and CMake, Rust, Julia ≥ 1.10, Python 3.10+
with PyTorch, Node 20+.

```bash
data/fetch.sh && python data/extract.py          # EuRoC MAV (~12 GB)
cmake -S vio -B vio/build && cmake --build vio/build -j && ctest --test-dir vio/build
(cd sim && cargo build --release && cargo test --release)
julia --project=mpc -e 'using Pkg; Pkg.test()'
eval/run_all.sh                                  # odometry + loops on the 11 sequences
eval/run_sim.sh                                  # simulated worlds 2-5
julia --project=mpc mpc/run_missions.jl 1 2 3 4 5
python perception/run_perception.py --train data/sim/world2 data/sim/world3 data/sim/world4 \
  --test data/sim/world5 --euroc data/euroc/V1_01_easy ... --out results/perception
python eval/report.py && python eval/export_viewer.py
(cd ui && npm ci && npm run build && npm run serve)   # http://localhost:8077
```

One sequence: `vio/build/rover-vio data/euroc/V1_01_easy --voc data/voc/rover.voc --out traj.txt --json result.json --loops loops.txt`.
Set `ROVER_NOSELFCAL=1` to turn off the stereo self-calibration.

The vocabulary `data/voc/rover.voc` (9883 words) was trained with `rover-voc`
on MH_01, V1_01 and V2_01. These are evaluation sequences too, a small
optimistic bias for loop detection.

## Tests and CI

- C++: 1102 checks. They cover Lie-group identities, numeric against analytic Jacobians, the sparse solver against dense, the pose graph, preintegration, the eigensolver and the PNG decoder (against zlib).
- Rust: 6 tests. Julia: 33. Python: 8. TypeScript: 9.
- GitHub Actions runs four jobs:
  - the C++ unit tests;
  - the simulator, then the odometry on a freshly generated world (ATE < 15 cm), then a full mission driven on the true pose and on the odometry (all goals, no collision);
  - perception on CPU;
  - the viewer.

## License

MIT
