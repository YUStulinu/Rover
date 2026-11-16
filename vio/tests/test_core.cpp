// Tests of the math, the sparse solver and the least-squares framework.
#include <cstdio>
#include <random>

#include "rover/factors.hpp"
#include "rover/imu.hpp"
#include "rover/reproj.hpp"
#include "rover/lsq.hpp"
#include "rover/math.hpp"
#include "rover/sparse.hpp"

using namespace rover;

int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(c)) {                                                           \
            ++g_failed;                                                       \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
        }                                                                     \
    } while (0)


static void test_lie() {
    std::mt19937 rng(1);
    std::normal_distribution<double> N(0, 1);
    for (int i = 0; i < 200; ++i) {
        Vec3 w{N(rng), N(rng), N(rng)};
        if (w.norm() > 3.0) continue;
        Mat3 R = so3_exp(w);
        CHECK((R * R.transpose() - Mat3::identity()).norm() < 1e-12);
        CHECK((so3_log(R) - w).norm() < 1e-9);
        Quat q = Quat::from_matrix(R);
        CHECK((q.matrix() - R).norm() < 1e-12);
        // the right Jacobian: exp(w + d) ~ exp(w) exp(Jr d)
        Vec3 d{1e-6, -2e-6, 3e-6};
        Mat3 lhs = so3_exp(w + d), rhs = R * so3_exp(so3_jr(w) * d);
        CHECK((lhs - rhs).norm() < 1e-10);
        CHECK((so3_jr(w) * so3_jr_inv(w) - Mat3::identity()).norm() < 1e-9);
    }
    // near pi
    Vec3 w{kPi - 1e-9, 0, 0};
    CHECK((so3_log(so3_exp(w)) - w).norm() < 1e-6);
}

static void test_sparse() {
    // a random sparse SPD matrix: L L^T + n I on a banded-plus-random pattern
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> U(-1, 1);
    int n = 300;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        A[i][i] = n * 0.2;
        for (int k = 0; k < 3; ++k) {
            int j = rng() % n;
            double v = U(rng);
            A[i][j] += v;
            A[j][i] += v;
        }
    }
    for (int i = 0; i < n; ++i) A[i][i] += 50;
    std::vector<std::tuple<int, int, double>> t;
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            if (A[i][j] != 0) t.emplace_back(i, j, A[i][j]);
    SparseSym S = SparseSym::from_triplets(n, t);
    std::vector<double> b(n);
    for (double& x : b) x = U(rng);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j && A[i][j] != 0) adj[i].push_back(j);
    std::vector<int> order = minimum_degree(adj);
    CHECK(static_cast<int>(order.size()) == n);
    LDLT ldl;
    CHECK(ldl.factor(S, order));
    std::vector<double> x = ldl.solve(b);
    std::vector<double> Ax = S.multiply(x);
    double err = 0;
    for (int i = 0; i < n; ++i) err = std::max(err, std::fabs(Ax[i] - b[i]));
    CHECK(err < 1e-10);
    // the identity order gives the same answer
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i) id[i] = i;
    LDLT ldl2;
    CHECK(ldl2.factor(S, id));
    std::vector<double> x2 = ldl2.solve(b);
    double d = 0;
    for (int i = 0; i < n; ++i) d = std::max(d, std::fabs(x[i] - x2[i]));
    CHECK(d < 1e-10);
    std::printf("  sparse: n %d, L nonzeros %zu with minimum degree, %zu without\n", n, ldl.nonzeros(), ldl2.nonzeros());
    // an indefinite matrix is refused
    std::vector<std::tuple<int, int, double>> t3 = {{0, 0, 1.0}, {0, 1, 2.0}, {1, 1, 1.0}};
    LDLT bad;
    CHECK(!bad.factor(SparseSym::from_triplets(2, t3), {0, 1}));
}

// a robot drives around a circle twice; odometry drifts; one loop closure
static void test_pose_graph() {
    std::mt19937 rng(3);
    std::normal_distribution<double> N(0, 1);
    const int K = 200;
    std::vector<SE3> truth(K);
    for (int k = 0; k < K; ++k) {
        double a = 2 * kPi * 2 * k / K;
        truth[k] = SE3(rot_z(a + kPi / 2) * so3_exp(Vec3{0.05, -0.03, 0}), Vec3{10 * std::cos(a), 10 * std::sin(a), 0.5 * std::sin(3 * a)});
    }
    for (int dof : {6, 4}) {
        Problem p;
        std::vector<Var*> v;
        std::vector<SE3> odo;
        for (int k = 0; k + 1 < K; ++k) {
            SE3 rel = truth[k].inverse() * truth[k + 1];
            // noise in yaw and position only for the 4-DoF case (roll and pitch known)
            Vec3 nr = dof == 6 ? Vec3{0.002 * N(rng), 0.002 * N(rng), 0.01 * N(rng)} : Vec3{0, 0, 0.01 * N(rng)};
            rel = SE3(rel.R * so3_exp(nr), rel.t + Vec3{0.02 * N(rng), 0.02 * N(rng), 0.01 * N(rng)});
            odo.push_back(rel);
        }
        std::vector<SE3> dead(K);
        dead[0] = truth[0];
        for (int k = 0; k + 1 < K; ++k) dead[k + 1] = dead[k] * odo[k];
        for (int k = 0; k < K; ++k) v.push_back(p.add(dof == 6 ? Var::pose(dead[k]) : Var::pose4(dead[k])));
        v[0]->fixed = true;
        for (int k = 0; k + 1 < K; ++k) {
            if (dof == 6)
                p.add_factor(std::make_unique<RelPoseFactor>(v[k], v[k + 1], odo[k], 0.01, 0.02));
            else
                p.add_factor(std::make_unique<RelPose4Factor>(v[k], v[k + 1], odo[k], 0.01, 0.02));
        }
        // loop closures: the second lap revisits the first
        for (int k = 0; k < K / 2; k += 10) {
            SE3 rel = truth[k].inverse() * truth[k + K / 2];
            if (dof == 6)
                p.add_factor(std::make_unique<RelPoseFactor>(v[k], v[k + K / 2], rel, 0.005, 0.01));
            else
                p.add_factor(std::make_unique<RelPose4Factor>(v[k], v[k + K / 2], rel, 0.005, 0.01));
        }
        auto err = [&]() {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (v[k]->pose_value().t - truth[k].t).squared_norm();
            return std::sqrt(s / K);
        };
        // what loop closure fixes: the second lap agrees with the first
        auto loop_err = [&]() {
            double s = 0;
            int n = 0;
            for (int k = 0; k < K / 2; ++k, ++n) {
                SE3 rel_est = v[k]->pose_value().inverse() * v[k + K / 2]->pose_value();
                SE3 rel_true = truth[k].inverse() * truth[k + K / 2];
                s += (rel_est.t - rel_true.t).squared_norm();
            }
            return std::sqrt(s / n);
        };
        double before = err(), loop_before = loop_err();
        SolveStats st = p.solve(30);
        double after = err(), loop_after = loop_err();
        std::printf("  pose graph %d-DoF: lap-to-lap error %.3f m -> %.3f m\n", dof, loop_before, loop_after);
        CHECK(loop_after < 0.08 && loop_after < loop_before / 10);
        CHECK(st.final_cost < 0.01 * st.initial_cost);
        std::printf("  pose graph %d-DoF: position RMSE %.3f m -> %.3f m in %d iterations (cost %.1f -> %.1f)\n", dof, before, after, st.iterations,
                    st.initial_cost, st.final_cost);
        CHECK(after < before);
    }
}

// analytic Jacobians of the reprojection factor against central differences
static void test_reproj_jacobians() {
    std::mt19937 rng(4);
    std::normal_distribution<double> N(0, 1);
    SE3 Tbc0(so3_exp(Vec3{0.01, -0.02, 1.57}), Vec3{-0.02, 0.06, 0.01}), Tbc1(so3_exp(Vec3{0.02, 0.01, 1.56}), Vec3{-0.02, -0.05, 0.0});
    for (int trial = 0; trial < 50; ++trial) {
        auto A = Var::pose(SE3(so3_exp(Vec3{0.1 * N(rng), 0.1 * N(rng), 0.3 * N(rng)}), Vec3{N(rng), N(rng), N(rng)}));
        auto T = Var::pose(SE3(A->pose_value().R * so3_exp(Vec3{0.05 * N(rng), 0.05 * N(rng), 0.05 * N(rng)}), A->pose_value().t + Vec3{0.2 * N(rng), 0.2 * N(rng), 0.2 * N(rng)}));
        double rho0 = 0.3 + 0.1 * std::fabs(N(rng));
        auto rho = Var::vector(1, &rho0);
        Vec2 xn{0.1 * N(rng), 0.1 * N(rng)};
        for (int same = 0; same < 2; ++same) {
            ReprojFactor f(same ? nullptr : A.get(), same ? nullptr : T.get(), rho.get(), xn, Vec2{0.01, -0.02}, Tbc0, same ? Tbc1 : Tbc0, 450.0, 0);
            struct Numeric : Factor {
                Factor* g;
                explicit Numeric(Factor* g_) : g(g_) {
                    vars = g->vars;
                    rdim = g->rdim;
                }
                void residual(double* r) const override { g->residual(r); }
            } num(&f);
            Problem p;
            std::vector<double> r1, r2;
            std::vector<std::vector<double>> J1, J2;
            p.factor_jacobians(f, r1, J1);
            p.factor_jacobians(num, r2, J2);
            double worst = 0, scale = 1e-9;
            for (size_t k = 0; k < J1.size(); ++k)
                for (size_t i = 0; i < J1[k].size(); ++i) {
                    worst = std::max(worst, std::fabs(J1[k][i] - J2[k][i]));
                    scale = std::max(scale, std::fabs(J2[k][i]));
                }
            CHECK(worst < 1e-5 * scale);
        }
    }
}

// a synthetic trajectory with exact IMU readings: the preintegrated residual vanishes at the truth
static void test_preintegration() {
    ImuNoise noise;
    Vec3 bg{0.01, -0.02, 0.005}, ba{0.05, -0.1, 0.08};
    auto pose_at = [](double t) {
        return SE3(so3_exp(Vec3{0.3 * std::sin(t), 0.2 * std::cos(1.3 * t), 0.5 * t}), Vec3{std::sin(t), std::cos(0.7 * t), 0.3 * t * t});
    };
    double h = 0.005, t0 = 1.0, t1 = 1.5;
    Preintegration pre(bg, ba, noise);
    auto vel = [&](double t) { return (pose_at(t + 1e-5).t - pose_at(t - 1e-5).t) * (1 / 2e-5); };
    for (double t = t0; t < t1 - 1e-9; t += h) {
        // the angular velocity in the body frame over this step, and the specific force at its middle
        SE3 a = pose_at(t), b = pose_at(t + h);
        Vec3 w = so3_log(a.R.transpose() * b.R) * (1 / h);
        double tm = t + h / 2;
        Vec3 acc_w = (pose_at(tm + 1e-4).t - 2.0 * pose_at(tm).t + pose_at(tm - 1e-4).t) * (1 / 1e-8);
        SE3 m = pose_at(tm);
        Vec3 f = m.R.transpose() * (acc_w - kGravity);
        // the IMU integrates with the attitude at the start of the step: correct for the half step
        f = so3_exp(w * (h / 2)) * f;
        pre.add(h, w + bg, f + ba);
    }
    pre.finish();
    double r[15];
    pre.residual(pose_at(t0), vel(t0), bg, ba, pose_at(t1), vel(t1), bg, ba, r);
    double worst = 0;
    for (int k = 0; k < 9; ++k) worst = std::max(worst, std::fabs(r[k]));
    std::printf("  preintegration residual at the truth (whitened): %.3f\n", worst);
    CHECK(worst < 3.0);
    // the bias Jacobians: compare with integrating again with other biases
    Vec3 dbg{0.002, -0.001, 0.003}, dba{0.02, 0.01, -0.03};
    Preintegration again = pre;
    again.repropagate(bg + dbg, ba + dba);
    Mat3 dRc = pre.dR * so3_exp(pre.JRg * dbg);
    Vec3 dvc = pre.dv + pre.Jvg * dbg + pre.Jva * dba, dpc = pre.dp + pre.Jpg * dbg + pre.Jpa * dba;
    CHECK(so3_log(dRc.transpose() * again.dR).norm() < 1e-5);
    CHECK((dvc - again.dv).norm() < 1e-4);
    CHECK((dpc - again.dp).norm() < 1e-4);
}

int main() {
    test_reproj_jacobians();
    test_preintegration();
    test_lie();
    test_sparse();
    test_pose_graph();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
