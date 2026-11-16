// Factors shared by the estimators: relative pose constraints (6 and 4
// degrees of freedom) and priors.
#pragma once

#include "lsq.hpp"

namespace rover {

inline double wrap_angle(double a) {
    while (a > kPi) a -= 2 * kPi;
    while (a < -kPi) a += 2 * kPi;
    return a;
}

/// T_i^-1 T_j should equal the measured relative motion (6 DoF).
struct RelPoseFactor : Factor {
    SE3 meas;
    double w_rot, w_pos;
    RelPoseFactor(Var* a, Var* b, const SE3& m, double sigma_rot, double sigma_pos) : meas(m), w_rot(1 / sigma_rot), w_pos(1 / sigma_pos) {
        vars = {a, b};
        rdim = 6;
    }
    void residual(double* r) const override {
        SE3 Ti = vars[0]->pose_value(), Tj = vars[1]->pose_value();
        SE3 rel = Ti.inverse() * Tj;
        SE3 e = meas.inverse() * rel;
        Vec3 dr = so3_log(e.R);
        for (int k = 0; k < 3; ++k) {
            r[k] = w_rot * dr[k];
            r[3 + k] = w_pos * e.t[k];
        }
    }
};

/// The 4-DoF version for visual-inertial pose graphs (roll and pitch are
/// observable from gravity, so only yaw and position drift): the relative
/// position in the frame of i and the relative yaw.
struct RelPose4Factor : Factor {
    Vec3 dp;
    double dyaw, w_yaw, w_pos;
    RelPose4Factor(Var* a, Var* b, const SE3& m_rel, double sigma_yaw, double sigma_pos)
        : dp(m_rel.t), dyaw(0), w_yaw(1 / sigma_yaw), w_pos(1 / sigma_pos) {
        vars = {a, b};
        rdim = 4;
        SE3 Ti = a->pose_value(), Tj = b->pose_value();
        // the measured yaw change, from the measured relative rotation and
        // the current roll and pitch of both ends
        dyaw = wrap_angle(yaw_of(Ti.R * m_rel.R) - yaw_of(Ti.R));
        (void)Tj;
    }
    void residual(double* r) const override {
        SE3 Ti = vars[0]->pose_value(), Tj = vars[1]->pose_value();
        Vec3 pij = Ti.R.transpose() * (Tj.t - Ti.t);
        for (int k = 0; k < 3; ++k) r[k] = w_pos * (pij[k] - dp[k]);
        r[3] = w_yaw * wrap_angle(yaw_of(Tj.R) - yaw_of(Ti.R) - dyaw);
    }
};

/// Keep a pose near a value.
struct PosePrior : Factor {
    SE3 meas;
    double w_rot, w_pos;
    PosePrior(Var* a, const SE3& m, double sigma_rot, double sigma_pos) : meas(m), w_rot(1 / sigma_rot), w_pos(1 / sigma_pos) {
        vars = {a};
        rdim = 6;
    }
    void residual(double* r) const override {
        SE3 e = meas.inverse() * vars[0]->pose_value();
        Vec3 dr = so3_log(e.R);
        for (int k = 0; k < 3; ++k) {
            r[k] = w_rot * dr[k];
            r[3 + k] = w_pos * e.t[k];
        }
    }
};

}  // namespace rover
