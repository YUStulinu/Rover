// Nonlinear least squares on manifolds: variables (poses, vectors), factors
// with whitened residuals, Jacobians by central differences in each
// variable's tangent space, Huber robust weights, Levenberg-Marquardt steps
// solved with the sparse LDL^T of sparse.hpp.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <memory>
#include <vector>

#include "math.hpp"
#include "sparse.hpp"

namespace rover {

enum class VarKind { Pose, Vector, PoseYawXYZ };

/// A variable. Pose: R (9, row-major) then p (3), tangent [dtheta (right), dp].
/// Vector: n numbers. PoseYawXYZ: like Pose but only yaw (about the world z
/// axis, applied on the left) and position move (4 dimensions).
struct Var {
    VarKind kind;
    int dim;
    double x[16];
    bool fixed = false;
    int index = -1;   // offset of its tangent in the system (set by the solver)
    int block = -1;

    static std::unique_ptr<Var> pose(const SE3& T) {
        auto v = std::make_unique<Var>();
        v->kind = VarKind::Pose;
        v->dim = 6;
        v->set_pose(T);
        return v;
    }
    static std::unique_ptr<Var> pose4(const SE3& T) {
        auto v = pose(T);
        v->kind = VarKind::PoseYawXYZ;
        v->dim = 4;
        return v;
    }
    static std::unique_ptr<Var> vector(int n, const double* init) {
        auto v = std::make_unique<Var>();
        v->kind = VarKind::Vector;
        v->dim = n;
        for (int i = 0; i < n; ++i) v->x[i] = init ? init[i] : 0.0;
        return v;
    }

    SE3 pose_value() const {
        SE3 T;
        for (int i = 0; i < 9; ++i) T.R[i] = x[i];
        for (int i = 0; i < 3; ++i) T.t[i] = x[9 + i];
        return T;
    }
    void set_pose(const SE3& T) {
        for (int i = 0; i < 9; ++i) x[i] = T.R[i];
        for (int i = 0; i < 3; ++i) x[9 + i] = T.t[i];
    }
    Vec3 vec3(int off = 0) const { return Vec3{x[off], x[off + 1], x[off + 2]}; }

    /// the tangent vector d with this = x0 (+) d
    void minus(const double* x0, double* d) const {
        switch (kind) {
            case VarKind::Pose: {
                Mat3 R0, R;
                for (int i = 0; i < 9; ++i) {
                    R0[i] = x0[i];
                    R[i] = x[i];
                }
                Vec3 w = so3_log(R0.transpose() * R);
                for (int k = 0; k < 3; ++k) {
                    d[k] = w[k];
                    d[3 + k] = x[9 + k] - x0[9 + k];
                }
                break;
            }
            case VarKind::PoseYawXYZ: {
                Mat3 R0, R;
                for (int i = 0; i < 9; ++i) {
                    R0[i] = x0[i];
                    R[i] = x[i];
                }
                double a = std::atan2(R(1, 0), R(0, 0)) - std::atan2(R0(1, 0), R0(0, 0));
                while (a > kPi) a -= 2 * kPi;
                while (a < -kPi) a += 2 * kPi;
                d[0] = a;
                for (int k = 0; k < 3; ++k) d[1 + k] = x[9 + k] - x0[9 + k];
                break;
            }
            case VarKind::Vector:
                for (int i = 0; i < dim; ++i) d[i] = x[i] - x0[i];
                break;
        }
    }

    void plus(const double* d) {
        switch (kind) {
            case VarKind::Pose: {
                SE3 T = pose_value();
                T.R = T.R * so3_exp(Vec3{d[0], d[1], d[2]});
                T.t += Vec3{d[3], d[4], d[5]};
                set_pose(T);
                break;
            }
            case VarKind::PoseYawXYZ: {
                SE3 T = pose_value();
                T.R = rot_z(d[0]) * T.R;
                T.t += Vec3{d[1], d[2], d[3]};
                set_pose(T);
                break;
            }
            case VarKind::Vector:
                for (int i = 0; i < dim; ++i) x[i] += d[i];
                break;
        }
    }
};

/// A factor: r(x) of dimension rdim, already whitened (multiplied by the
/// square root of its information), with an optional Huber threshold.
struct Factor {
    std::vector<Var*> vars;
    int rdim = 0;
    double huber = 0;  // 0: plain least squares
    virtual ~Factor() = default;
    virtual void residual(double* r) const = 0;
    /// analytic Jacobians: return false to use central differences
    virtual bool jacobians(std::vector<std::vector<double>>& /*J*/) const { return false; }
    bool active = true;
};

struct SolveStats {
    int iterations = 0;
    double initial_cost = 0, final_cost = 0;
    bool converged = false;
    size_t factor_nonzeros = 0;
};

class Problem {
public:
    std::vector<std::unique_ptr<Var>> owned;
    std::vector<Var*> vars;
    std::vector<std::unique_ptr<Factor>> factors;

    Var* add(std::unique_ptr<Var> v) {
        owned.push_back(std::move(v));
        vars.push_back(owned.back().get());
        return vars.back();
    }
    /// a variable that lives elsewhere (it must outlive the problem's use)
    Var* add_external(Var* v) {
        vars.push_back(v);
        return v;
    }
    template <class F>
    F* add_factor(std::unique_ptr<F> f) {
        F* p = f.get();
        factors.push_back(std::move(f));
        return p;
    }

    /// robust cost: sum of rho(|r|^2) / 2
    double cost() const {
        double c = 0;
        std::vector<double> r;
        for (auto& f : factors) {
            if (!f->active) continue;
            r.assign(f->rdim, 0.0);
            f->residual(r.data());
            double s = 0;
            for (double v : r) s += v * v;
            c += 0.5 * rho(s, f->huber);
        }
        return c;
    }

    SolveStats solve(int max_iterations = 20, double tol = 1e-8, bool verbose = false) {
        SolveStats st;
        // index the free variables
        int n = 0, nb = 0;
        std::vector<int> offset, dims;
        for (Var* v : vars) {
            if (v->fixed) {
                v->index = -1;
                v->block = -1;
                continue;
            }
            v->index = n;
            v->block = nb++;
            offset.push_back(n);
            dims.push_back(v->dim);
            n += v->dim;
        }
        if (n == 0) return st;
        // ordering from the graph of variables
        std::vector<std::vector<int>> adj(nb);
        for (auto& f : factors)
            for (Var* a : f->vars)
                for (Var* b : f->vars)
                    if (a != b && a->block >= 0 && b->block >= 0) adj[a->block].push_back(b->block);
        std::vector<int> perm = expand_order(minimum_degree(adj), offset, dims);

        double lambda = 1e-4;
        double cost0 = cost();
        st.initial_cost = cost0;
        std::vector<double> saved;
        LDLT ldl;
        bool analyzed = false;
        for (int it = 0; it < max_iterations; ++it) {
            std::vector<double> g(n, 0.0);
            SparseSym H0 = assemble(g, offset, dims);
            if (!analyzed) {
                ldl.analyze(H0, perm);
                analyzed = true;
            }
            // positions of the diagonal entries
            std::vector<int> dpos(n, -1);
            for (int c = 0; c < n; ++c)
                for (int k = H0.colptr[c]; k < H0.colptr[c + 1]; ++k)
                    if (H0.rowidx[k] == c) dpos[c] = k;
            bool improved = false;
            for (int attempt = 0; attempt < 10; ++attempt) {
                SparseSym H = H0;
                for (int i = 0; i < n; ++i) H.val[dpos[i]] += lambda * std::max(H0.val[dpos[i]], 1e-9) + 1e-12;
                bool ok = ldl.refactor(H);
                st.factor_nonzeros = ldl.nonzeros();
                if (!ok) {
                    lambda *= 10;
                    continue;
                }
                std::vector<double> dx = ldl.solve(g);  // H dx = -J^T r  (g holds -J^T r)
                save(saved);
                apply(dx);
                double c = cost();
                if (c < cost0) {
                    double rel = (cost0 - c) / std::max(cost0, 1e-30);
                    cost0 = c;
                    lambda = std::max(lambda / 3, 1e-9);
                    improved = true;
                    st.iterations = it + 1;
                    if (verbose) std::printf("  iteration %d: cost %.6g, lambda %.1e\n", it, c, lambda);
                    if (rel < tol) st.converged = true;
                    break;
                }
                restore(saved);
                lambda *= 10;
            }
            if (!improved) {
                st.converged = true;
                break;
            }
            if (st.converged) break;
        }
        st.final_cost = cost0;
        return st;
    }

    static double rho(double s, double huber) {
        if (huber <= 0 || s <= huber * huber) return s;
        return 2 * huber * std::sqrt(s) - huber * huber;
    }
    static double rho_weight(double s, double huber) {
        if (huber <= 0 || s <= huber * huber) return 1.0;
        return huber / std::sqrt(s);
    }

    /// Jacobians of one factor with respect to each of its free variables
    /// (central differences unless the factor provides them).
    void factor_jacobians(Factor& f, std::vector<double>& r, std::vector<std::vector<double>>& J) const {
        r.assign(f.rdim, 0.0);
        f.residual(r.data());
        J.assign(f.vars.size(), {});
        for (size_t k = 0; k < f.vars.size(); ++k) J[k].assign(static_cast<size_t>(f.rdim) * f.vars[k]->dim, 0.0);
        if (f.jacobians(J)) return;
        std::vector<double> rp(f.rdim), rm(f.rdim);
        for (size_t k = 0; k < f.vars.size(); ++k) {
            Var* v = f.vars[k];
            if (v->fixed) continue;
            double save_x[16];
            std::memcpy(save_x, v->x, sizeof save_x);
            for (int d = 0; d < v->dim; ++d) {
                double h = 1e-6;
                double step[16] = {0};
                step[d] = h;
                v->plus(step);
                f.residual(rp.data());
                std::memcpy(v->x, save_x, sizeof save_x);
                step[d] = -h;
                v->plus(step);
                f.residual(rm.data());
                std::memcpy(v->x, save_x, sizeof save_x);
                for (int i = 0; i < f.rdim; ++i) J[k][i * v->dim + d] = (rp[i] - rm[i]) / (2 * h);
            }
        }
    }

private:
    /// J^T J (upper triangle, with the full diagonal blocks) and -J^T r,
    /// accumulated in dense blocks per pair of variables, then laid out as
    /// compressed columns. The pattern only depends on which variables share
    /// a factor, so it is the same at every iteration.
    SparseSym assemble(std::vector<double>& g, const std::vector<int>& offset, const std::vector<int>& dims) const {
        struct Block {
            int rb, cb;
            std::vector<double> v;  // dims[rb] x dims[cb], row-major
        };
        std::unordered_map<uint64_t, size_t> where;
        std::vector<Block> blocks;
        int nb = static_cast<int>(dims.size());
        for (int b = 0; b < nb; ++b) {  // every diagonal block exists
            where[(uint64_t(b) << 32) | uint64_t(b)] = blocks.size();
            blocks.push_back({b, b, std::vector<double>(static_cast<size_t>(dims[b]) * dims[b], 0.0)});
        }
        std::vector<double> r;
        std::vector<std::vector<double>> J;
        for (auto& fp : factors) {
            Factor& f = *fp;
            if (!f.active) continue;
            factor_jacobians(f, r, J);
            double s = 0;
            for (double v : r) s += v * v;
            double w = rho_weight(s, f.huber);
            for (size_t a = 0; a < f.vars.size(); ++a) {
                Var* va = f.vars[a];
                if (va->fixed) continue;
                for (int i = 0; i < va->dim; ++i) {
                    double acc = 0;
                    for (int k = 0; k < f.rdim; ++k) acc += J[a][k * va->dim + i] * r[k];
                    g[va->index + i] -= w * acc;
                }
                for (size_t b = 0; b < f.vars.size(); ++b) {
                    Var* vb = f.vars[b];
                    if (vb->fixed || vb->block < va->block) continue;
                    uint64_t key = (uint64_t(va->block) << 32) | uint64_t(vb->block);
                    auto it = where.find(key);
                    size_t bi;
                    if (it == where.end()) {
                        bi = blocks.size();
                        where[key] = bi;
                        blocks.push_back({va->block, vb->block, std::vector<double>(static_cast<size_t>(va->dim) * vb->dim, 0.0)});
                    } else {
                        bi = it->second;
                    }
                    double* out = blocks[bi].v.data();
                    for (int i = 0; i < va->dim; ++i)
                        for (int j = 0; j < vb->dim; ++j) {
                            double acc = 0;
                            for (int k = 0; k < f.rdim; ++k) acc += J[a][k * va->dim + i] * J[b][k * vb->dim + j];
                            out[i * vb->dim + j] += w * acc;
                        }
                }
            }
        }
        // columns: blocks grouped by their column block, sorted by row block
        std::vector<std::vector<size_t>> bycol(nb);
        for (size_t q = 0; q < blocks.size(); ++q) bycol[blocks[q].cb].push_back(q);
        int n = offset.empty() ? 0 : offset.back() + dims.back();
        SparseSym H;
        H.n = n;
        H.colptr.assign(n + 1, 0);
        for (int cb = 0; cb < nb; ++cb) {
            auto& list = bycol[cb];
            std::sort(list.begin(), list.end(), [&](size_t x, size_t y) { return blocks[x].rb < blocks[y].rb; });
            for (int j = 0; j < dims[cb]; ++j) {
                int col = offset[cb] + j;
                for (size_t q : list) {
                    const Block& B = blocks[q];
                    int rows = dims[B.rb];
                    int lim = B.rb == cb ? j + 1 : rows;  // upper triangle inside a diagonal block
                    for (int i = 0; i < lim; ++i) {
                        H.rowidx.push_back(offset[B.rb] + i);
                        H.val.push_back(B.v[static_cast<size_t>(i) * dims[cb] + j]);
                    }
                }
                H.colptr[col + 1] = static_cast<int>(H.rowidx.size());
            }
        }
        return H;
    }

    void save(std::vector<double>& s) const {
        s.clear();
        for (Var* v : vars)
            if (!v->fixed) s.insert(s.end(), v->x, v->x + 16);
    }
    void restore(const std::vector<double>& s) {
        size_t k = 0;
        for (Var* v : vars)
            if (!v->fixed) {
                std::memcpy(v->x, &s[k], 16 * sizeof(double));
                k += 16;
            }
    }
    void apply(const std::vector<double>& dx) {
        for (Var* v : vars)
            if (!v->fixed) v->plus(&dx[v->index]);
    }
};

}  // namespace rover
