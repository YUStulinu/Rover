#include "rover/estimator.hpp"
#include "rover/factors.hpp"
#include "rover/reproj.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <set>

namespace rover {

void MarginalPrior::residual(double* r) const {
    std::vector<double> d;
    int n = J.c;
    d.reserve(n);
    for (size_t k = 0; k < vars.size(); ++k) {
        double dk[16];
        vars[k]->minus(x0[k].data(), dk);
        for (int i = 0; i < vars[k]->dim; ++i) d.push_back(dk[i]);
    }
    static const double scale = std::getenv("ROVER_PRIORSCALE") ? std::atof(std::getenv("ROVER_PRIORSCALE")) : 1.0;
    for (int i = 0; i < rdim; ++i) {
        double s = r0[i];
        for (int j = 0; j < n; ++j) s += J(i, j) * d[j];
        r[i] = s * scale;
    }
}

Estimator::Estimator(const Camera& left, const Camera& right, const ImuNoise& noise, const EstimatorConfig& cfg) : noise_(noise), cfg_(cfg) {
    cam_[0] = left;
    cam_[1] = right;
}

Vec3 Estimator::landmark_world(const Landmark& l) const {
    double rho = std::max(l.rho->x[0], 1e-6);
    Vec3 Xc{l.xn[0] / rho, l.xn[1] / rho, 1.0 / rho};
    return l.anchor->pose->pose_value() * (cam_[0].T_bc * Xc);
}

std::vector<Vec3> Estimator::landmark_points() const {
    std::vector<Vec3> out;
    for (auto& [id, l] : lm_) out.push_back(landmark_world(l));
    return out;
}

std::vector<std::pair<double, SE3>> Estimator::window_poses() const {
    std::vector<std::pair<double, SE3>> out;
    for (auto& f : frames_) out.push_back({f->t, f->pose->pose_value()});
    return out;
}

bool Estimator::add_frame(double t, const std::vector<ImuSample>& imu, const std::unordered_map<int, Observation>& obs) {
    outliers_.clear();
    auto f = std::make_unique<FrameState>();
    f->t = t;
    f->index = frame_count_++;
    f->obs = obs;
    double zero[6] = {0, 0, 0, 0, 0, 0};
    if (!initialized_) {
        // attitude from gravity: the mean specific force points up in the world
        Vec3 a;
        int n = 0;
        for (auto& s : imu) {
            a += s.acc;
            ++n;
        }
        if (n < 10) return false;
        a = a * (1.0 / n);
        Vec3 up = a * (1.0 / a.norm()), z{0, 0, 1};
        Vec3 axis = cross(up, z);
        double s = axis.norm(), c = dot(up, z);
        Mat3 R = s < 1e-9 ? Mat3::identity() : so3_exp(axis * (std::atan2(s, c) / s));
        f->pose = Var::pose(SE3(R, Vec3{}));
        initial_pose_ = SE3(R, Vec3{});
        initial_prior_ = true;
        f->vel = Var::vector(3, zero);
        f->bias = Var::vector(6, zero);
        f->pre = Preintegration(Vec3{}, Vec3{}, noise_);
        f->keyframe = true;
        frames_.push_back(std::move(f));
        initialized_ = true;
    } else {
        FrameState& prev = *frames_.back();
        Vec3 bg = prev.bias->vec3(0), ba = prev.bias->vec3(3);
        f->pre = Preintegration(bg, ba, noise_);
        for (auto& s : imu) f->pre.add(s.t, s.gyro, s.acc);
        f->pre.finish();
        // predict with the IMU
        SE3 Ti = prev.pose->pose_value();
        Vec3 vi = prev.vel->vec3();
        double dt = f->pre.dt;
        SE3 Tj(orthonormalize(Ti.R * f->pre.dR), Ti.t + vi * dt + 0.5 * dt * dt * kGravity + Ti.R * f->pre.dp);
        Vec3 vj = vi + kGravity * dt + Ti.R * f->pre.dv;
        f->pose = Var::pose(Tj);
        f->vel = Var::vector(3, &vj[0]);
        double b[6] = {bg[0], bg[1], bg[2], ba[0], ba[1], ba[2]};
        f->bias = Var::vector(6, b);
        // keyframe: enough parallax or few features in common with the last keyframe
        const FrameState* kf = nullptr;
        for (auto it = frames_.rbegin(); it != frames_.rend(); ++it)
            if ((*it)->keyframe) {
                kf = it->get();
                break;
            }
        int common = 0;
        double par = 0;
        for (auto& [id, o] : obs) {
            auto it = kf->obs.find(id);
            if (it == kf->obs.end()) continue;
            ++common;
            par += (o.uv[0] - it->second.uv[0]).norm();
        }
        f->keyframe = common < cfg_.keyframe_min_common || (common > 0 && par / common > cfg_.keyframe_parallax_px);
        frames_.push_back(std::move(f));
    }
    FrameState* cur = frames_.back().get();
    // new landmarks from stereo
    SE3 T_rl = cam_[1].T_bc.inverse() * cam_[0].T_bc;
    for (auto& [id, o] : cur->obs) {
        if (lm_.count(id) || !o.stereo) continue;
        double d = triangulate_depth(o.xn[0], o.xn[1], T_rl);
        if (!(d > cfg_.min_depth && d < cfg_.max_depth)) continue;
        Landmark l;
        l.anchor = cur;
        l.xn = o.xn[0];
        double rho = 1.0 / d;
        l.rho = Var::vector(1, &rho);
        lm_[id] = std::move(l);
    }
    if (frames_.size() >= 2) {
        optimize();
        remove_outliers();
    }
    slide();
    return true;
}

void Estimator::build(Problem& p, bool with_prior) {
    double w = cam_[0].fx / cfg_.pixel_sigma, hub = cfg_.huber_px;
    for (size_t i = 0; i < frames_.size(); ++i) {
        FrameState& f = *frames_[i];
        p.add_external(f.pose.get());
        p.add_external(f.vel.get());
        p.add_external(f.bias.get());
        f.pose->fixed = false;
        if (i > 0) {
            FrameState& g = *frames_[i - 1];
            p.add_factor(std::make_unique<ImuFactor>(g.pose.get(), g.vel.get(), g.bias.get(), f.pose.get(), f.vel.get(), f.bias.get(), &f.pre));
        }
    }
    for (auto& [id, l] : lm_) {
        int n_obs = 0;
        for (auto& fp : frames_) {
            FrameState& f = *fp;
            auto it = f.obs.find(id);
            if (it == f.obs.end()) continue;
            const Observation& o = it->second;
            if (&f == l.anchor) {
                if (o.stereo)
                    p.add_factor(std::make_unique<ReprojFactor>(nullptr, nullptr, l.rho.get(), l.xn, o.xn[1], cam_[0].T_bc, cam_[1].T_bc, w, hub)), ++n_obs;
                continue;
            }
            p.add_factor(std::make_unique<ReprojFactor>(l.anchor->pose.get(), f.pose.get(), l.rho.get(), l.xn, o.xn[0], cam_[0].T_bc, cam_[0].T_bc, w, hub));
            ++n_obs;
            if (o.stereo) {
                p.add_factor(std::make_unique<ReprojFactor>(l.anchor->pose.get(), f.pose.get(), l.rho.get(), l.xn, o.xn[1], cam_[0].T_bc, cam_[1].T_bc, w, hub));
                ++n_obs;
            }
        }
        if (n_obs > 0) p.add_external(l.rho.get());
    }
    if (!prior_ && initial_prior_) {
        // the gauge: the first frame's pose is known (position and yaw are not observable)
        p.add_factor(std::make_unique<PosePrior>(frames_[0]->pose.get(), initial_pose_, 1e-3, 1e-3));
    }
    if (with_prior && prior_ && !std::getenv("ROVER_NOPRIOR")) {
        // the prior is owned by the estimator: wrap it so that the problem does not delete it
        struct Ref : Factor {
            Factor* f;
            explicit Ref(Factor* f_) : f(f_) {
                vars = f->vars;
                rdim = f->rdim;
            }
            void residual(double* r) const override { f->residual(r); }
        };
        p.add_factor(std::make_unique<Ref>(prior_.get()));
    }
}

void Estimator::optimize() {
    Problem p;
    build(p, true);
    SolveStats st = p.solve(cfg_.iterations, 1e-6);
    last_cost = st.final_cost;
    last_iterations = st.iterations;
    // integrate the IMU data again where a bias moved a lot
    for (size_t i = 1; i < frames_.size(); ++i) {
        FrameState& f = *frames_[i];
        FrameState& g = *frames_[i - 1];
        Vec3 bg = g.bias->vec3(0), ba = g.bias->vec3(3);
        if ((bg - f.pre.bg).norm() > 0.01 || (ba - f.pre.ba).norm() > 0.1) {
            f.pre.repropagate(bg, ba);
            f.pre.finish();
        }
    }
}

void Estimator::remove_outliers() {
    double w = cam_[0].fx / cfg_.pixel_sigma;
    std::vector<int> bad_lm;
    for (auto& [id, l] : lm_) {
        double rho = l.rho->x[0];
        if (!(rho > 1.0 / cfg_.max_depth && rho < 1.0 / cfg_.min_depth)) {
            bad_lm.push_back(id);
            continue;
        }
        for (auto& fp : frames_) {
            FrameState& f = *fp;
            auto it = f.obs.find(id);
            if (it == f.obs.end()) continue;
            for (int k = 0; k < 2; ++k) {
                if (k == 1 && !it->second.stereo) continue;
                if (&f == l.anchor && k == 0) continue;
                ReprojFactor rf(&f == l.anchor ? nullptr : l.anchor->pose.get(), f.pose.get(), l.rho.get(), l.xn, it->second.xn[k], cam_[0].T_bc,
                                cam_[k].T_bc, w, 0);
                double r[2];
                rf.residual(r);
                double e = std::sqrt(r[0] * r[0] + r[1] * r[1]) * cfg_.pixel_sigma;
                if (e > cfg_.max_reprojection_px) {
                    if (k == 1)
                        it->second.stereo = false;
                    else if (&f == frames_.back().get()) {
                        outliers_.push_back(id);
                        f.obs.erase(it);
                        break;
                    } else {
                        f.obs.erase(it);
                        break;
                    }
                }
            }
        }
    }
    for (int id : bad_lm) {
        lm_.erase(id);
        outliers_.push_back(id);
    }
}

void Estimator::reanchor(int id, Landmark& l) {
    Vec3 Xw = landmark_world(l);
    for (auto& fp : frames_) {
        FrameState& f = *fp;
        if (&f == l.anchor) continue;
        auto it = f.obs.find(id);
        if (it == f.obs.end()) continue;
        Vec3 Xc = (f.pose->pose_value() * cam_[0].T_bc).inverse() * Xw;
        if (Xc[2] < cfg_.min_depth) continue;
        l.anchor = &f;
        l.xn = it->second.xn[0];
        l.rho->x[0] = 1.0 / Xc[2];
        return;
    }
    l.anchor = nullptr;
}

void Estimator::slide() {
    if (static_cast<int>(frames_.size()) <= cfg_.window) return;
    if (!frames_[frames_.size() - 2]->keyframe)
        drop_second_newest();
    else
        marginalize_oldest();
}

void Estimator::drop_second_newest() {
    size_t k = frames_.size() - 2;
    FrameState* drop = frames_[k].get();
    FrameState& next = *frames_[k + 1];
    FrameState& before = *frames_[k - 1];
    // the IMU data of both intervals, integrated again from the earlier frame's biases
    Preintegration merged(before.bias->vec3(0), before.bias->vec3(3), noise_);
    for (auto& s : drop->pre.samples) merged.add(s.t, s.gyro, s.acc);
    for (auto& s : next.pre.samples) merged.add(s.t, s.gyro, s.acc);
    merged.finish();
    next.pre = merged;
    std::vector<int> dead;
    for (auto& [id, l] : lm_)
        if (l.anchor == drop) {
            reanchor(id, l);
            if (!l.anchor) dead.push_back(id);
        }
    for (int id : dead) lm_.erase(id);
    frames_.erase(frames_.begin() + static_cast<long>(k));
}

void Estimator::marginalize_oldest() {
    FrameState* old = frames_.front().get();
    FrameState& second = *frames_[1];
    double w = cam_[0].fx / cfg_.pixel_sigma, hub = cfg_.huber_px;
    // the factors that touch what is marginalized: the velocity and biases of
    // the oldest frame (pose, velocity, biases) and the landmarks anchored in it
    std::vector<std::unique_ptr<Factor>> fs;
    old->pose->fixed = false;
    if (!prior_ && initial_prior_) fs.push_back(std::make_unique<PosePrior>(old->pose.get(), initial_pose_, 1e-3, 1e-3));
    fs.push_back(std::make_unique<ImuFactor>(old->pose.get(), old->vel.get(), old->bias.get(), second.pose.get(), second.vel.get(), second.bias.get(),
                                             &second.pre));
    std::vector<Var*> marg = {old->pose.get(), old->vel.get(), old->bias.get()};
    for (auto& [id, l] : lm_) {
        if (l.anchor != old) continue;
        bool used = false;
        for (auto& fp : frames_) {
            FrameState& f = *fp;
            if (!f.keyframe || &f == frames_.back().get()) continue;  // the prior stays on keyframes
            auto it = f.obs.find(id);
            if (it == f.obs.end()) continue;
            if (&f == old) {
                if (it->second.stereo)
                    fs.push_back(std::make_unique<ReprojFactor>(nullptr, nullptr, l.rho.get(), l.xn, it->second.xn[1], cam_[0].T_bc, cam_[1].T_bc, w, hub)),
                        used = true;
                continue;
            }
            fs.push_back(std::make_unique<ReprojFactor>(old->pose.get(), f.pose.get(), l.rho.get(), l.xn, it->second.xn[0], cam_[0].T_bc, cam_[0].T_bc, w, hub));
            if (it->second.stereo)
                fs.push_back(std::make_unique<ReprojFactor>(old->pose.get(), f.pose.get(), l.rho.get(), l.xn, it->second.xn[1], cam_[0].T_bc, cam_[1].T_bc, w, hub));
            used = true;
        }
        if (used) marg.push_back(l.rho.get());
    }
    struct Ref : Factor {
        Factor* f;
        explicit Ref(Factor* f_) : f(f_) {
            vars = f->vars;
            rdim = f->rdim;
        }
        void residual(double* r) const override { f->residual(r); }
    };
    if (prior_) fs.push_back(std::make_unique<Ref>(prior_.get()));
    // the kept variables: everything else these factors touch (not fixed)
    std::vector<Var*> keep;
    std::set<Var*> mset(marg.begin(), marg.end());
    for (auto& f : fs)
        for (Var* v : f->vars)
            if (!v->fixed && !mset.count(v) && std::find(keep.begin(), keep.end(), v) == keep.end()) keep.push_back(v);
    // a variable of the oldest frame can appear in the old prior only through vel/bias (marginalized)
    int m = 0, n = 0;
    for (Var* v : marg) {
        v->index = m;
        m += v->dim;
    }
    for (Var* v : keep) {
        v->index = m + n;
        n += v->dim;
    }
    int N = m + n;
    MatX H(N, N);
    std::vector<double> b(N, 0.0);
    Problem helper;
    std::vector<double> r;
    std::vector<std::vector<double>> J;
    for (auto& fp : fs) {
        Factor& f = *fp;
        helper.factor_jacobians(f, r, J);
        double s = 0;
        for (double v : r) s += v * v;
        double wgt = Problem::rho_weight(s, f.huber);
        for (size_t a = 0; a < f.vars.size(); ++a) {
            Var* va = f.vars[a];
            if (va->fixed) continue;
            for (int i = 0; i < va->dim; ++i) {
                double acc = 0;
                for (int k = 0; k < f.rdim; ++k) acc += J[a][k * va->dim + i] * r[k];
                b[va->index + i] -= wgt * acc;
            }
            for (size_t c = 0; c < f.vars.size(); ++c) {
                Var* vc = f.vars[c];
                if (vc->fixed) continue;
                for (int i = 0; i < va->dim; ++i)
                    for (int j = 0; j < vc->dim; ++j) {
                        double acc = 0;
                        for (int k = 0; k < f.rdim; ++k) acc += J[a][k * va->dim + i] * J[c][k * vc->dim + j];
                        H(va->index + i, vc->index + j) += wgt * acc;
                    }
            }
        }
    }
    // Schur complement onto the kept variables
    MatX Hmm(m, m), Hmk(m, n), Hkk(n, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) Hmm(i, j) = H(i, j);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) Hmk(i, j) = H(i, m + j);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Hkk(i, j) = H(m + i, m + j);
    MatX Hinv = spd_inverse(Hmm);
    MatX T = matmul(transpose(Hmk), Hinv);  // n x m
    MatX S = matmul(T, Hmk);
    std::vector<double> bk(n);
    for (int i = 0; i < n; ++i) {
        double s = b[m + i];
        for (int k = 0; k < m; ++k) s -= T(i, k) * b[k];
        bk[i] = s;
        for (int j = 0; j < n; ++j) Hkk(i, j) -= S(i, j);
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) Hkk(i, j) = Hkk(j, i) = 0.5 * (Hkk(i, j) + Hkk(j, i));
    // H = J^T J, b = -J^T r0
    std::vector<double> ev;
    MatX V;
    sym_eigen(Hkk, ev, V);
    auto pr = std::make_unique<MarginalPrior>();
    pr->vars = keep;
    for (Var* v : keep) {
        std::array<double, 16> x;
        std::memcpy(x.data(), v->x, sizeof(double) * 16);
        pr->x0.push_back(x);
    }
    double emax = *std::max_element(ev.begin(), ev.end());
    std::vector<int> rows;
    for (int i = 0; i < n; ++i)
        if (ev[i] > std::max(1e-8, emax * 1e-12)) rows.push_back(i);
    pr->rdim = static_cast<int>(rows.size());
    pr->J = MatX(pr->rdim, n);
    pr->r0.assign(pr->rdim, 0.0);
    for (size_t q = 0; q < rows.size(); ++q) {
        int i = rows[q];
        double s = std::sqrt(ev[i]);
        double proj = 0;
        for (int k = 0; k < n; ++k) {
            pr->J(static_cast<int>(q), k) = s * V(k, i);
            proj += V(k, i) * bk[k];
        }
        pr->r0[q] = -proj / s;
    }
    prior_ = std::move(pr);
    initial_prior_ = false;
    // remove the oldest frame and its landmarks
    std::vector<int> dead;
    for (auto& [id, l] : lm_)
        if (l.anchor == old) dead.push_back(id);
    for (int id : dead) lm_.erase(id);
    frames_.pop_front();
}

}  // namespace rover
