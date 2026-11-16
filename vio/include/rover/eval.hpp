// Trajectory accuracy: the absolute trajectory error after aligning the
// estimate to the ground truth with a rigid motion (Horn's closed form via
// an SVD-free quaternion method), and the relative pose error over fixed
// distances travelled.
#pragma once

#include <vector>

#include "dense.hpp"
#include "math.hpp"

namespace rover {

struct Stamped {
    double t;
    SE3 T;
};

/// Rigid alignment (no scale) minimizing sum |R a_i + t - b_i|^2 (Horn 1987, quaternion form).
inline SE3 align_rigid(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    size_t n = a.size();
    Vec3 ca, cb;
    for (size_t i = 0; i < n; ++i) {
        ca += a[i];
        cb += b[i];
    }
    ca = ca * (1.0 / n);
    cb = cb * (1.0 / n);
    Mat3 S;
    for (size_t i = 0; i < n; ++i) {
        Vec3 x = a[i] - ca, y = b[i] - cb;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) S(r, c) += x[r] * y[c];
    }
    MatX N(4, 4);
    double sxx = S(0, 0), sxy = S(0, 1), sxz = S(0, 2), syx = S(1, 0), syy = S(1, 1), syz = S(1, 2), szx = S(2, 0), szy = S(2, 1), szz = S(2, 2);
    double vals[16] = {sxx + syy + szz, syz - szy,        szx - sxz,        sxy - syx,        syz - szy, sxx - syy - szz, sxy + syx,         szx + sxz,
                       szx - sxz,       sxy + syx,        -sxx + syy - szz, syz + szy,        sxy - syx, szx + sxz,       syz + szy,         -sxx - syy + szz};
    for (int i = 0; i < 16; ++i) N.a[i] = vals[i];
    std::vector<double> w;
    MatX V;
    sym_eigen(N, w, V);
    int best = 0;
    for (int i = 1; i < 4; ++i)
        if (w[i] > w[best]) best = i;
    Quat q(V(0, best), V(1, best), V(2, best), V(3, best));
    Mat3 R = q.normalized().matrix();
    return SE3(R, cb - R * ca);
}

struct TrajectoryError {
    double ate_rmse = 0, ate_mean = 0, ate_max = 0;
    double rpe_trans_pct = 0;  // mean translation error over segments, % of their length
    double rpe_rot_deg_per_m = 0;
    double length = 0;
    int matched = 0;
    SE3 alignment;
};

/// est and gt sampled at the same times (gt interpolated by the caller).
inline TrajectoryError trajectory_error(const std::vector<Stamped>& est, const std::vector<Stamped>& gt, const std::vector<double>& segments = {1, 2, 4, 8}) {
    TrajectoryError e;
    size_t n = std::min(est.size(), gt.size());
    e.matched = static_cast<int>(n);
    if (n < 3) return e;
    std::vector<Vec3> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = est[i].T.t;
        b[i] = gt[i].T.t;
    }
    e.alignment = align_rigid(a, b);
    double s2 = 0, s = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (e.alignment * a[i] - b[i]).norm();
        s2 += d * d;
        s += d;
        e.ate_max = std::max(e.ate_max, d);
    }
    e.ate_rmse = std::sqrt(s2 / n);
    e.ate_mean = s / n;
    // distance along the ground truth
    std::vector<double> dist(n, 0.0);
    for (size_t i = 1; i < n; ++i) dist[i] = dist[i - 1] + (gt[i].T.t - gt[i - 1].T.t).norm();
    e.length = dist[n - 1];
    double tsum = 0, rsum = 0;
    int cnt = 0;
    for (double seg : segments) {
        size_t j = 0;
        for (size_t i = 0; i < n; i += 5) {
            while (j < n && dist[j] - dist[i] < seg) ++j;
            if (j >= n) break;
            SE3 de = est[i].T.inverse() * est[j].T, dg = gt[i].T.inverse() * gt[j].T;
            SE3 err = dg.inverse() * de;
            tsum += err.t.norm() / seg;
            rsum += so3_log(err.R).norm() * 180 / kPi / seg;
            ++cnt;
        }
    }
    if (cnt) {
        e.rpe_trans_pct = 100 * tsum / cnt;
        e.rpe_rot_deg_per_m = rsum / cnt;
    }
    return e;
}

}  // namespace rover
