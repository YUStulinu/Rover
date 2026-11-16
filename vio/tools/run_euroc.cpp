// rover-vio: stereo visual-inertial odometry with loop closure on an EuRoC sequence.
//
//   rover-vio SEQUENCE_DIR [--voc VOCABULARY] [--out FILE] [--frames N] [--json OUT]
//             [--loops OUT]
//
// Writes the trajectory (TUM format: time x y z qx qy qz qw, the body frame;
// after the final pose-graph optimization when loops are closed) and prints
// the absolute trajectory error against the ground truth: of the odometry
// alone, of the poses as they were output (corrected by the loops known at
// the time), and of the final trajectory.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "rover/estimator.hpp"
#include "rover/euroc.hpp"
#include "rover/eval.hpp"
#include "rover/frontend.hpp"
#include "rover/loop.hpp"

using namespace rover;

static TrajectoryError against_truth(const EurocSequence& seq, const std::vector<Stamped>& traj) {
    std::vector<Stamped> e2, g2;
    for (auto& s : traj) {
        SE3 g;
        if (seq.truth_at(s.t, g)) {
            e2.push_back(s);
            g2.push_back({s.t, g});
        }
    }
    return trajectory_error(e2, g2);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: rover-vio SEQUENCE_DIR [--voc FILE] [--out FILE] [--frames N] [--json OUT] [--loops OUT]\n");
        return 2;
    }
    std::string dir = argv[1], out_path, json_path, voc_path, loops_path;
    struct LoopRecord {
        double ti, tj;
        int good;
    };
    std::vector<LoopRecord> loop_records;
    size_t max_frames = ~size_t(0);
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--out") out_path = argv[i + 1];
        else if (a == "--frames") max_frames = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--json") json_path = argv[i + 1];
        else if (a == "--voc") voc_path = argv[i + 1];
        else if (a == "--loops") loops_path = argv[i + 1];
    }
    EurocSequence seq = EurocSequence::load(dir);
    FrontendConfig fcfg;
    fcfg.self_calibrate = std::getenv("ROVER_NOSELFCAL") == nullptr;  // for the ablation
    Frontend fe(seq.cam[0], seq.cam[1], fcfg);
    Estimator est(seq.cam[0], seq.cam[1], seq.noise);
    Vocabulary voc;
    bool loops_on = !voc_path.empty();
    if (loops_on && !voc.load(voc_path)) {
        std::fprintf(stderr, "cannot read vocabulary %s\n", voc_path.c_str());
        return 1;
    }
    LoopDetector det(&voc);
    PoseGraph pg;
    std::vector<Stamped> odom, online;
    std::vector<int> kf_of;  // the latest keyframe at each output frame (-1: none yet)
    int loops_found = 0, loops_correct = 0;
    double loop_ms = 0;
    size_t imu_i = 0;
    double t_prev = -1;
    auto t0 = std::chrono::steady_clock::now();
    double fe_time = 0, be_time = 0;
    size_t n = std::min(max_frames, seq.frames.size());
    for (size_t k = 0; k < n; ++k) {
        const Frame& fr = seq.frames[k];
        auto a = std::chrono::steady_clock::now();
        Image L = read_png(fr.left), R = read_png(fr.right);
        const auto& obs = fe.process(L, R);
        auto b = std::chrono::steady_clock::now();
        // the IMU between the previous frame and this one, piecewise constant
        // (the mean of each pair of samples over the part of their interval
        // inside [t_prev, t])
        std::vector<ImuSample> imu;
        if (t_prev < 0) {
            for (auto& s : seq.imu)
                if (std::fabs(s.t - fr.t) < 0.25) imu.push_back(s);
        } else {
            while (imu_i + 1 < seq.imu.size() && seq.imu[imu_i + 1].t <= t_prev) ++imu_i;
            for (size_t i = imu_i; i + 1 < seq.imu.size() && seq.imu[i].t < fr.t; ++i) {
                double s0 = std::max(seq.imu[i].t, t_prev), s1 = std::min(seq.imu[i + 1].t, fr.t);
                if (s1 <= s0) continue;
                imu.push_back({s1 - s0, 0.5 * (seq.imu[i].gyro + seq.imu[i + 1].gyro), 0.5 * (seq.imu[i].acc + seq.imu[i + 1].acc)});
            }
        }
        bool ok = est.add_frame(fr.t, imu, obs);
        fe.reject(est.outliers());
        auto c = std::chrono::steady_clock::now();
        fe_time += std::chrono::duration<double>(b - a).count();
        be_time += std::chrono::duration<double>(c - b).count();
        if (!ok) continue;
        t_prev = fr.t;
        SE3 T = est.pose();
        if (loops_on && est.last_is_keyframe()) {
            auto la = std::chrono::steady_clock::now();
            LoopKeyframe kf;
            kf.t = fr.t;
            kf.T_wb = T;
            kf.orb = loop_features(L);
            std::vector<Vec2> pix;
            for (auto& o : kf.orb) pix.push_back(Vec2{o.x, o.y});
            std::vector<double> depth = fe.stereo_depths(L, R, pix);
            for (size_t i = 0; i < pix.size(); ++i) {
                Vec2 xn = seq.cam[0].unproject(pix[i]);
                kf.pts.push_back(depth[i] > 0 ? Vec3{xn[0] * depth[i], xn[1] * depth[i], depth[i]} : Vec3{0, 0, -1});
            }
            pg.add_keyframe(fr.t, T);
            LoopConstraint lc;
            if (det.add(std::move(kf), seq.cam[0].T_bc, lc)) {
                ++loops_found;
                // is it right? compare with the ground truth
                SE3 gi, gj;
                bool good = false;
                if (seq.truth_at(pg.time(lc.i), gi) && seq.truth_at(pg.time(lc.j), gj)) {
                    SE3 e = (gi.inverse() * gj).inverse() * lc.T_ij;
                    good = e.t.norm() < 0.3 && so3_log(e.R).norm() < 5 * kPi / 180;
                }
                loops_correct += good;
                loop_records.push_back({pg.time(lc.i), pg.time(lc.j), good ? 1 : 0});
                pg.add_loop(lc);
                pg.optimize();
            }
            loop_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - la).count();
        }
        odom.push_back({fr.t, T});
        int kf = pg.size() ? static_cast<int>(pg.size()) - 1 : -1;
        kf_of.push_back(kf);
        online.push_back({fr.t, kf >= 0 ? pg.correction(kf) * T : T});
        if (k % (std::getenv("ROVER_DEBUG") ? 20 : 500) == 0)
            std::fprintf(stderr, "frame %zu t %.1f tracked %d landmarks %zu keyframes %zu loops %d pos %.2f %.2f %.2f |v| %.2f |bg| %.3f |ba| %.3f\n", k, fr.t,
                         fe.tracked_last(), est.landmarks(),
                         pg.size(), loops_found, T.t[0], T.t[1], T.t[2], est.velocity().norm(), est.gyro_bias().norm(), est.acc_bias().norm());
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // the final trajectory: every pose moved with the final correction of its keyframe
    std::vector<Stamped> final_traj;
    for (size_t i = 0; i < odom.size(); ++i) final_traj.push_back({odom[i].t, kf_of[i] >= 0 ? pg.correction(kf_of[i]) * odom[i].T : odom[i].T});
    if (!out_path.empty()) {
        std::FILE* out = std::fopen(out_path.c_str(), "w");
        for (auto& s : final_traj) {
            Quat q = Quat::from_matrix(s.T.R);
            std::fprintf(out, "%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", s.t + seq.t0_ns * 1e-9, s.T.t[0], s.T.t[1], s.T.t[2], q.x, q.y, q.z, q.w);
        }
        std::fclose(out);
    }
    if (!loops_path.empty()) {  // the closed loops: the times of both keyframes, and whether each was right
        std::FILE* lf = std::fopen(loops_path.c_str(), "w");
        for (auto& r : loop_records) std::fprintf(lf, "%.9f %.9f %d\n", r.ti + seq.t0_ns * 1e-9, r.tj + seq.t0_ns * 1e-9, r.good);
        std::fclose(lf);
    }
    TrajectoryError eo = against_truth(seq, odom), en = against_truth(seq, online), ef = against_truth(seq, final_traj);
    char json[2048];
    std::snprintf(json, sizeof json,
                  "{\"sequence\": \"%s\", \"frames\": %zu, \"length_m\": %.2f, \"ate_odometry_m\": %.4f, \"ate_online_m\": %.4f, \"ate_final_m\": %.4f, "
                  "\"ate_max_final_m\": %.4f, \"rpe_trans_pct\": %.3f, \"rpe_rot_deg_per_m\": %.4f, \"keyframes\": %zu, \"loops\": %d, \"loops_correct\": %d, "
                  "\"loop_candidates\": %d, \"seconds\": %.1f, \"frontend_ms\": %.2f, \"backend_ms\": %.2f, \"loop_ms_per_keyframe\": %.2f, "
                  "\"stereo_row_offset_px\": %.2f}",
                  dir.c_str(), odom.size(), eo.length, eo.ate_rmse, en.ate_rmse, ef.ate_rmse, ef.ate_max, eo.rpe_trans_pct, eo.rpe_rot_deg_per_m, pg.size(),
                  loops_found, loops_correct, det.candidates_checked, secs, 1000 * fe_time / std::max<size_t>(n, 1), 1000 * be_time / std::max<size_t>(n, 1),
                  pg.size() ? loop_ms / pg.size() : 0.0, fe.stereo_row_offset());
    std::printf("%s\n", json);
    if (!json_path.empty()) {
        std::FILE* j = std::fopen(json_path.c_str(), "w");
        if (j) {
            std::fputs(json, j);
            std::fclose(j);
        }
    }
    return 0;
}
