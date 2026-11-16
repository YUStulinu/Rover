// The reprojection factor of a landmark stored as inverse depth along a
// ray of its anchor frame's left camera, observed in camera k of a target
// frame (or in the right camera of the anchor itself).
#pragma once

#include "lsq.hpp"

namespace rover {

/// Reprojection of a landmark (inverse depth rho along xn in the anchor's
/// left camera) into camera k of a target frame.
struct ReprojFactor : Factor {
    Vec2 xn, obs;
    SE3 T_bc0, T_bck;
    double w;
    bool same;  // the target is the anchor (only the stereo observation)
    SE3 T_ck_c0;
    ReprojFactor(Var* anchor, Var* target, Var* rho, const Vec2& xn_, const Vec2& obs_, const SE3& T_bc0_, const SE3& T_bck_, double w_, double huber_)
        : xn(xn_), obs(obs_), T_bc0(T_bc0_), T_bck(T_bck_), w(w_), same(anchor == nullptr) {
        if (same) {
            vars = {rho};
            T_ck_c0 = T_bck.inverse() * T_bc0;
        } else {
            vars = {anchor, target, rho};
        }
        rdim = 2;
        huber = huber_;
    }
    void residual(double* r) const override {
        double rho = vars.back()->x[0];
        if (rho < 1e-6) rho = 1e-6;
        Vec3 Xc0{xn[0] / rho, xn[1] / rho, 1.0 / rho};
        Vec3 Xk;
        if (same) {
            Xk = T_ck_c0 * Xc0;
        } else {
            SE3 Ta = vars[0]->pose_value(), Tt = vars[1]->pose_value();
            Vec3 Xw = Ta * (T_bc0 * Xc0);
            Xk = (Tt * T_bck).inverse() * Xw;
        }
        if (Xk[2] < 1e-3) {
            r[0] = r[1] = 1e3;
            return;
        }
        r[0] = w * (Xk[0] / Xk[2] - obs[0]);
        r[1] = w * (Xk[1] / Xk[2] - obs[1]);
    }
    // analytic Jacobians: right perturbations of the poses, d(inverse depth)
    bool jacobians(std::vector<std::vector<double>>& J) const override {
        double rho = vars.back()->x[0];
        if (rho < 1e-6) rho = 1e-6;
        Vec3 f{xn[0], xn[1], 1.0};
        Vec3 Xc0 = f * (1.0 / rho);
        Vec3 Xk;
        Mat<3, 1> dXk_drho;
        Mat3 dXk_dthA, dXk_dpA, dXk_dthT, dXk_dpT;
        if (same) {
            Xk = T_ck_c0 * Xc0;
            dXk_drho = T_ck_c0.R * (f * (-1.0 / (rho * rho)));
        } else {
            SE3 Ta = vars[0]->pose_value(), Tt = vars[1]->pose_value();
            Vec3 Xb = T_bc0 * Xc0;
            Vec3 Xw = Ta * Xb;
            Mat3 RtT = Tt.R.transpose(), RkT = T_bck.R.transpose();
            Vec3 Xbt = RtT * (Xw - Tt.t);
            Xk = RkT * (Xbt - T_bck.t);
            Mat3 A = RkT * RtT;  // dXk / dXw
            dXk_dthA = A * (-(Ta.R * skew(Xb)));
            dXk_dpA = A;
            dXk_dthT = RkT * skew(Xbt);
            dXk_dpT = -A;
            dXk_drho = A * (Ta.R * (T_bc0.R * (f * (-1.0 / (rho * rho)))));
        }
        if (Xk[2] < 1e-3) return false;
        double iz = 1.0 / Xk[2];
        Mat<2, 3> P{w * iz, 0, -w * Xk[0] * iz * iz, 0, w * iz, -w * Xk[1] * iz * iz};
        auto put = [&](size_t k, const Mat<2, 3>& M, int col0, int dim) {
            for (int rr = 0; rr < 2; ++rr)
                for (int c = 0; c < 3; ++c) J[k][rr * dim + col0 + c] = M(rr, c);
        };
        if (same) {
            Mat<2, 1> d = P * dXk_drho;
            J[0][0] = d[0];
            J[0][1] = d[1];
            return true;
        }
        put(0, P * dXk_dthA, 0, 6);
        put(0, P * dXk_dpA, 3, 6);
        put(1, P * dXk_dthT, 0, 6);
        put(1, P * dXk_dpT, 3, 6);
        Mat<2, 1> d = P * dXk_drho;
        J[2][0] = d[0];
        J[2][1] = d[1];
        return true;
    }
};


}  // namespace rover
