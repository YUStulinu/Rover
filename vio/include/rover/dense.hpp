// Dense matrices of any size: products, Cholesky solves and a symmetric
// eigendecomposition (cyclic Jacobi), for marginalization.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace rover {

struct MatX {
    int r = 0, c = 0;
    std::vector<double> a;
    MatX() = default;
    MatX(int r_, int c_) : r(r_), c(c_), a(static_cast<size_t>(r_) * c_, 0.0) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(i) * c + j]; }
    double operator()(int i, int j) const { return a[static_cast<size_t>(i) * c + j]; }
};

inline MatX matmul(const MatX& x, const MatX& y) {
    MatX m(x.r, y.c);
    for (int i = 0; i < x.r; ++i)
        for (int k = 0; k < x.c; ++k) {
            double v = x(i, k);
            if (v == 0) continue;
            for (int j = 0; j < y.c; ++j) m(i, j) += v * y(k, j);
        }
    return m;
}

inline MatX transpose(const MatX& x) {
    MatX t(x.c, x.r);
    for (int i = 0; i < x.r; ++i)
        for (int j = 0; j < x.c; ++j) t(j, i) = x(i, j);
    return t;
}

/// Inverse of a symmetric positive definite matrix via Cholesky; adds a
/// tiny regularization so that a nearly singular block stays invertible.
inline MatX spd_inverse(MatX A) {
    int n = A.r;
    for (int i = 0; i < n; ++i) A(i, i) += 1e-9;
    MatX L(n, n);
    for (int j = 0; j < n; ++j) {
        double s = A(j, j);
        for (int k = 0; k < j; ++k) s -= L(j, k) * L(j, k);
        L(j, j) = std::sqrt(std::max(s, 1e-12));
        for (int i = j + 1; i < n; ++i) {
            double t = A(i, j);
            for (int k = 0; k < j; ++k) t -= L(i, k) * L(j, k);
            L(i, j) = t / L(j, j);
        }
    }
    // invert L, then A^-1 = L^-T L^-1
    MatX Li(n, n);
    for (int i = 0; i < n; ++i) {
        Li(i, i) = 1.0 / L(i, i);
        for (int j = 0; j < i; ++j) {
            double s = 0;
            for (int k = j; k < i; ++k) s += L(i, k) * Li(k, j);
            Li(i, j) = -s / L(i, i);
        }
    }
    MatX inv(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0;
            for (int k = i; k < n; ++k) s += Li(k, i) * Li(k, j);
            inv(i, j) = inv(j, i) = s;
        }
    return inv;
}

/// Eigendecomposition of a symmetric matrix: A = V diag(w) V^T (columns of V).
inline void sym_eigen(MatX A, std::vector<double>& w, MatX& V) {
    int n = A.r;
    V = MatX(n, n);
    for (int i = 0; i < n; ++i) V(i, i) = 1;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += A(p, q) * A(p, q);
        if (off < 1e-22) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                double apq = A(p, q);
                if (std::fabs(apq) < 1e-300) continue;
                double theta = (A(q, q) - A(p, p)) / (2 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; ++k) {
                    double akp = A(k, p), akq = A(k, q);
                    A(k, p) = c * akp - s * akq;
                    A(k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    double apk = A(p, k), aqk = A(q, k);
                    A(p, k) = c * apk - s * aqk;
                    A(q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    double vkp = V(k, p), vkq = V(k, q);
                    V(k, p) = c * vkp - s * vkq;
                    V(k, q) = s * vkp + c * vkq;
                }
            }
    }
    w.resize(n);
    for (int i = 0; i < n; ++i) w[i] = A(i, i);
}

}  // namespace rover
