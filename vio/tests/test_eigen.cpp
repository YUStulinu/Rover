// The Jacobi eigendecomposition and the SPD inverse used in marginalization.
#include <cmath>
#include <cstdio>
#include <random>
#include "rover/dense.hpp"
using namespace rover;
int main() {
    int fails = 0;
    std::mt19937 rng(5);
    std::normal_distribution<double> N(0, 1);
    for (int n : {4, 30, 120}) {
        MatX A(n, n), M(n, n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) M(i, j) = N(rng);
        A = matmul(transpose(M), M);
        std::vector<double> w; MatX V;
        sym_eigen(A, w, V);
        double err = 0, orth = 0;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            double s = 0, o = 0;
            for (int k = 0; k < n; ++k) { s += V(i, k) * w[k] * V(j, k); o += V(k, i) * V(k, j); }
            err = std::max(err, std::fabs(s - A(i, j)));
            orth = std::max(orth, std::fabs(o - (i == j)));
        }
        MatX Ai = spd_inverse(A), P = matmul(A, Ai);
        double inv = 0;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) inv = std::max(inv, std::fabs(P(i, j) - (i == j)));
        std::printf("n %d: reconstruction %.2e, orthogonality %.2e, inverse %.2e\n", n, err, orth, inv);
        // A = M^T M of a square Gaussian M is badly conditioned (about n^4), hence the loose inverse bound
        if (!(err < 1e-9 && orth < 1e-12 && inv < 1e-4)) ++fails;
    }
    std::printf("%s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
