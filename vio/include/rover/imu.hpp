// IMU preintegration on the manifold (Forster, Carlone, Dellaert and
// Scaramuzza, "On-Manifold Preintegration for Real-Time Visual-Inertial
// Odometry", T-RO 2017): the rotation, velocity and position increments
// between two frames, their first-order corrections for bias changes, and
// their covariance; and the factor that ties two frames' states together.
#pragma once

#include <vector>

#include "camera.hpp"
#include "euroc.hpp"
#include "lsq.hpp"
#include "math.hpp"

namespace rover {

inline const Vec3 kGravity{0, 0, -9.81};

/// Cholesky L L^T of a small symmetric positive definite matrix (lower L).
template <int N>
Mat<N, N> cholesky(const Mat<N, N>& A) {
    Mat<N, N> L;
    for (int j = 0; j < N; ++j) {
        double s = A(j, j);
        for (int k = 0; k < j; ++k) s -= L(j, k) * L(j, k);
        L(j, j) = std::sqrt(std::max(s, 1e-300));
        for (int i = j + 1; i < N; ++i) {
            double t = A(i, j);
            for (int k = 0; k < j; ++k) t -= L(i, k) * L(j, k);
            L(i, j) = t / L(j, j);
        }
    }
    return L;
}

class Preintegration {
public:
    Vec3 bg, ba;  // the biases the increments were integrated with
    double dt = 0;
    Mat3 dR = Mat3::identity();
    Vec3 dv, dp;
    Mat3 JRg, Jvg, Jva, Jpg, Jpa;
    Mat<9, 9> cov;
    ImuNoise noise;
    std::vector<ImuSample> samples;  // kept to integrate again with new biases

    Preintegration() = default;
    Preintegration(const Vec3& bg_, const Vec3& ba_, const ImuNoise& n) : bg(bg_), ba(ba_), noise(n) {}

    /// one IMU interval of length h with the measurements at its start
    void add(double h, const Vec3& gyro, const Vec3& acc, bool keep = true) {
        if (keep) samples.push_back({h, gyro, acc});
        Vec3 w = gyro - bg, a = acc - ba;
        Mat3 dRk = so3_exp(w * h), Jr = so3_jr(w * h);
        Mat3 Ra = dR * skew(a);
        // bias Jacobians (first order, in the order that uses the old values)
        Jpa += Jva * h - 0.5 * h * h * dR;
        Jpg += Jvg * h - 0.5 * h * h * (Ra * JRg);
        Jva -= h * dR;
        Jvg -= h * (Ra * JRg);
        JRg = dRk.transpose() * JRg - Jr * h;
        // covariance of [dtheta, dv, dp]
        Mat<9, 9> A = Mat<9, 9>::identity();
        A.set_block<3, 3>(0, 0, dRk.transpose());
        A.set_block<3, 3>(3, 0, -h * Ra);
        A.set_block<3, 3>(6, 0, -0.5 * h * h * Ra);
        A.set_block<3, 3>(6, 3, h * Mat3::identity());
        Mat<9, 6> B;
        B.set_block<3, 3>(0, 0, Jr * h);
        B.set_block<3, 3>(3, 3, h * dR);
        B.set_block<3, 3>(6, 3, 0.5 * h * h * dR);
        Mat<6, 6> Q;
        double gn = noise.gyro_noise * noise.gyro_noise / h, an = noise.acc_noise * noise.acc_noise / h;
        for (int k = 0; k < 3; ++k) {
            Q(k, k) = gn;
            Q(3 + k, 3 + k) = an;
        }
        cov = A * cov * A.transpose() + B * Q * B.transpose();
        // the increments
        dp += dv * h + 0.5 * h * h * (dR * a);
        dv += h * (dR * a);
        dR = orthonormalize(dR * dRk);
        dt += h;
    }

    /// integrate everything again with new linearization biases
    void repropagate(const Vec3& bg_, const Vec3& ba_) {
        auto s = samples;
        *this = Preintegration(bg_, ba_, noise);
        for (auto& x : s) add(x.t, x.gyro, x.acc);
    }

    /// append the samples of the next interval (when a frame between is dropped)
    void merge(const Preintegration& next) {
        for (auto& x : next.samples) add(x.t, x.gyro, x.acc);
    }

    /// the 15 residuals between two states (whitened)
    void residual(const SE3& Ti, const Vec3& vi, const Vec3& bgi, const Vec3& bai, const SE3& Tj, const Vec3& vj, const Vec3& bgj, const Vec3& baj,
                  double* r) const {
        Vec3 dbg = bgi - bg, dba = bai - ba;
        Mat3 dRc = dR * so3_exp(JRg * dbg);
        Vec3 dvc = dv + Jvg * dbg + Jva * dba;
        Vec3 dpc = dp + Jpg * dbg + Jpa * dba;
        Mat3 RiT = Ti.R.transpose();
        Vec3 eR = so3_log(dRc.transpose() * RiT * Tj.R);
        Vec3 ev = RiT * (vj - vi - kGravity * dt) - dvc;
        Vec3 ep = RiT * (Tj.t - Ti.t - vi * dt - 0.5 * dt * dt * kGravity) - dpc;
        Mat<9, 1> e9;
        for (int k = 0; k < 3; ++k) {
            e9[k] = eR[k];
            e9[3 + k] = ev[k];
            e9[6 + k] = ep[k];
        }
        Mat<9, 1> w = sqrt_info_.transpose() * e9;
        for (int k = 0; k < 9; ++k) r[k] = w[k];
        double sg = noise.gyro_walk * std::sqrt(std::max(dt, 1e-3)), sa = noise.acc_walk * std::sqrt(std::max(dt, 1e-3));
        for (int k = 0; k < 3; ++k) {
            r[9 + k] = (bgj[k] - bgi[k]) / sg;
            r[12 + k] = (baj[k] - bai[k]) / sa;
        }
    }

    /// compute the square root of the information once the interval is complete
    void finish() {
        Mat<9, 9> c = cov;
        for (int k = 0; k < 9; ++k) c(k, k) += 1e-12;
        sqrt_info_ = cholesky(inverse(c));
    }

private:
    Mat<9, 9> sqrt_info_ = Mat<9, 9>::identity();
};

/// Ties pose i, velocity i, biases i (gyro then accelerometer, one 6-vector)
/// to pose j, velocity j, biases j.
struct ImuFactor : Factor {
    const Preintegration* pre;
    ImuFactor(Var* Ti, Var* vi, Var* bi, Var* Tj, Var* vj, Var* bj, const Preintegration* p) : pre(p) {
        vars = {Ti, vi, bi, Tj, vj, bj};
        rdim = 15;
    }
    void residual(double* r) const override {
        pre->residual(vars[0]->pose_value(), vars[1]->vec3(), vars[2]->vec3(0), vars[2]->vec3(3), vars[3]->pose_value(), vars[4]->vec3(), vars[5]->vec3(0),
                      vars[5]->vec3(3), r);
    }
};

}  // namespace rover
