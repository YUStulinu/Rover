// Small fixed-size linear algebra, rotations (SO3) and rigid motions (SE3).
// Matrices are row-major; everything is double precision.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>

namespace rover {

constexpr double kPi = 3.14159265358979323846;

template <int R, int C>
struct Mat {
    std::array<double, R * C> a{};

    Mat() = default;
    Mat(std::initializer_list<double> v) {
        int i = 0;
        for (double x : v) a[i++] = x;
    }
    static Mat zero() { return Mat(); }
    static Mat identity() {
        static_assert(R == C);
        Mat m;
        for (int i = 0; i < R; ++i) m(i, i) = 1;
        return m;
    }
    double& operator()(int r, int c) { return a[r * C + c]; }
    double operator()(int r, int c) const { return a[r * C + c]; }
    double& operator[](int i) { return a[i]; }
    double operator[](int i) const { return a[i]; }

    Mat& operator+=(const Mat& o) {
        for (int i = 0; i < R * C; ++i) a[i] += o.a[i];
        return *this;
    }
    Mat& operator-=(const Mat& o) {
        for (int i = 0; i < R * C; ++i) a[i] -= o.a[i];
        return *this;
    }
    Mat& operator*=(double s) {
        for (double& x : a) x *= s;
        return *this;
    }
    friend Mat operator+(Mat x, const Mat& y) { return x += y; }
    friend Mat operator-(Mat x, const Mat& y) { return x -= y; }
    friend Mat operator*(Mat x, double s) { return x *= s; }
    friend Mat operator*(double s, Mat x) { return x *= s; }
    friend Mat operator-(Mat x) { return x *= -1.0; }

    Mat<C, R> transpose() const {
        Mat<C, R> t;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) t(c, r) = (*this)(r, c);
        return t;
    }
    template <int R2, int C2>
    Mat<R2, C2> block(int r0, int c0) const {
        Mat<R2, C2> b;
        for (int r = 0; r < R2; ++r)
            for (int c = 0; c < C2; ++c) b(r, c) = (*this)(r0 + r, c0 + c);
        return b;
    }
    template <int R2, int C2>
    void set_block(int r0, int c0, const Mat<R2, C2>& b) {
        for (int r = 0; r < R2; ++r)
            for (int c = 0; c < C2; ++c) (*this)(r0 + r, c0 + c) = b(r, c);
    }
    double squared_norm() const {
        double s = 0;
        for (double x : a) s += x * x;
        return s;
    }
    double norm() const { return std::sqrt(squared_norm()); }
};

template <int R, int K, int C>
Mat<R, C> operator*(const Mat<R, K>& x, const Mat<K, C>& y) {
    Mat<R, C> m;
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < K; ++k) {
            double v = x(r, k);
            if (v == 0) continue;
            for (int c = 0; c < C; ++c) m(r, c) += v * y(k, c);
        }
    return m;
}

using Vec2 = Mat<2, 1>;
using Vec3 = Mat<3, 1>;
using Vec6 = Mat<6, 1>;
using Mat2 = Mat<2, 2>;
using Mat3 = Mat<3, 3>;
using Mat6 = Mat<6, 6>;

inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Mat3 skew(const Vec3& v) { return Mat3{0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}; }

/// Inverse of a symmetric positive definite (or any invertible) small matrix by Gauss-Jordan.
template <int N>
Mat<N, N> inverse(Mat<N, N> m) {
    Mat<N, N> inv = Mat<N, N>::identity();
    for (int c = 0; c < N; ++c) {
        int p = c;
        for (int r = c + 1; r < N; ++r)
            if (std::fabs(m(r, c)) > std::fabs(m(p, c))) p = r;
        if (p != c)
            for (int k = 0; k < N; ++k) {
                std::swap(m(p, k), m(c, k));
                std::swap(inv(p, k), inv(c, k));
            }
        double d = m(c, c);
        for (int k = 0; k < N; ++k) {
            m(c, k) /= d;
            inv(c, k) /= d;
        }
        for (int r = 0; r < N; ++r) {
            if (r == c) continue;
            double f = m(r, c);
            if (f == 0) continue;
            for (int k = 0; k < N; ++k) {
                m(r, k) -= f * m(c, k);
                inv(r, k) -= f * inv(c, k);
            }
        }
    }
    return inv;
}

// ---- rotations --------------------------------------------------------------

struct Quat {
    double w = 1, x = 0, y = 0, z = 0;
    Quat() = default;
    Quat(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}
    Quat operator*(const Quat& q) const {
        return {w * q.w - x * q.x - y * q.y - z * q.z, w * q.x + x * q.w + y * q.z - z * q.y, w * q.y - x * q.z + y * q.w + z * q.x,
                w * q.z + x * q.y - y * q.x + z * q.w};
    }
    Quat conj() const { return {w, -x, -y, -z}; }
    Quat normalized() const {
        double n = std::sqrt(w * w + x * x + y * y + z * z);
        return {w / n, x / n, y / n, z / n};
    }
    Mat3 matrix() const {
        return Mat3{1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                    2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                    2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
    }
    static Quat from_matrix(const Mat3& R) {
        double tr = R(0, 0) + R(1, 1) + R(2, 2);
        Quat q;
        if (tr > 0) {
            double s = std::sqrt(tr + 1.0) * 2;
            q = {0.25 * s, (R(2, 1) - R(1, 2)) / s, (R(0, 2) - R(2, 0)) / s, (R(1, 0) - R(0, 1)) / s};
        } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
            double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2;
            q = {(R(2, 1) - R(1, 2)) / s, 0.25 * s, (R(0, 1) + R(1, 0)) / s, (R(0, 2) + R(2, 0)) / s};
        } else if (R(1, 1) > R(2, 2)) {
            double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2;
            q = {(R(0, 2) - R(2, 0)) / s, (R(0, 1) + R(1, 0)) / s, 0.25 * s, (R(1, 2) + R(2, 1)) / s};
        } else {
            double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2;
            q = {(R(1, 0) - R(0, 1)) / s, (R(0, 2) + R(2, 0)) / s, (R(1, 2) + R(2, 1)) / s, 0.25 * s};
        }
        return q.normalized();
    }
};

/// exp: rotation vector -> rotation matrix (Rodrigues).
inline Mat3 so3_exp(const Vec3& w) {
    double th = w.norm();
    Mat3 K = skew(w);
    if (th < 1e-8) return Mat3::identity() + K + 0.5 * (K * K);
    return Mat3::identity() + (std::sin(th) / th) * K + ((1 - std::cos(th)) / (th * th)) * (K * K);
}

/// log: rotation matrix -> rotation vector.
inline Vec3 so3_log(const Mat3& R) {
    double c = std::fmax(-1.0, std::fmin(1.0, (R(0, 0) + R(1, 1) + R(2, 2) - 1) / 2));
    double th = std::acos(c);
    Vec3 v{R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1)};
    if (th < 1e-8) return 0.5 * v;
    if (kPi - th < 1e-6) {
        // near pi: from the symmetric part
        Quat q = Quat::from_matrix(R);
        Vec3 axis{q.x, q.y, q.z};
        double n = axis.norm();
        return (th / (n > 0 ? n : 1)) * axis;
    }
    return (th / (2 * std::sin(th))) * v;
}

/// Right Jacobian of SO3 and its inverse.
inline Mat3 so3_jr(const Vec3& w) {
    double th = w.norm();
    Mat3 K = skew(w);
    if (th < 1e-8) return Mat3::identity() - 0.5 * K;
    return Mat3::identity() - ((1 - std::cos(th)) / (th * th)) * K + ((th - std::sin(th)) / (th * th * th)) * (K * K);
}
inline Mat3 so3_jr_inv(const Vec3& w) {
    double th = w.norm();
    Mat3 K = skew(w);
    if (th < 1e-8) return Mat3::identity() + 0.5 * K;
    return Mat3::identity() + 0.5 * K + (1 / (th * th) - (1 + std::cos(th)) / (2 * th * std::sin(th))) * (K * K);
}

/// Re-orthonormalize a rotation matrix.
inline Mat3 orthonormalize(const Mat3& R) { return Quat::from_matrix(R).matrix(); }

/// A rigid motion: x -> R x + t.
struct SE3 {
    Mat3 R = Mat3::identity();
    Vec3 t;
    SE3() = default;
    SE3(const Mat3& R_, const Vec3& t_) : R(R_), t(t_) {}
    SE3 operator*(const SE3& o) const { return {R * o.R, R * o.t + t}; }
    Vec3 operator*(const Vec3& p) const { return R * p + t; }
    SE3 inverse() const {
        Mat3 Rt = R.transpose();
        return {Rt, -(Rt * t)};
    }
};

inline double yaw_of(const Mat3& R) { return std::atan2(R(1, 0), R(0, 0)); }
inline Mat3 rot_z(double a) { return Mat3{std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1}; }

}  // namespace rover
