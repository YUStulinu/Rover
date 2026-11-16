#include "rover/loop.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

#include "rover/eval.hpp"
#include "rover/factors.hpp"

namespace rover {

namespace {
Desc majority(const std::vector<const Desc*>& set) {
    Desc c{};
    if (set.empty()) return c;
    for (int b = 0; b < 256; ++b) {
        size_t ones = 0;
        for (const Desc* d : set) ones += ((*d)[b >> 6] >> (b & 63)) & 1;
        if (2 * ones > set.size()) c[b >> 6] |= uint64_t(1) << (b & 63);
    }
    return c;
}
}  // namespace

void Vocabulary::build(int node, std::vector<const Desc*>& set, int k, int level, int levels, uint32_t& seed) {
    if (level == levels || static_cast<int>(set.size()) <= k) {
        nodes_[node].word = n_words_++;
        return;
    }
    // k-majority clustering, seeded k-means++ style by farthest-first sampling
    std::mt19937 rng(seed++);
    std::vector<Desc> centers;
    centers.push_back(*set[rng() % set.size()]);
    std::vector<int> dmin(set.size(), 1 << 30);
    while (static_cast<int>(centers.size()) < k) {
        long total = 0;
        for (size_t i = 0; i < set.size(); ++i) {
            dmin[i] = std::min(dmin[i], hamming(*set[i], centers.back()));
            total += static_cast<long>(dmin[i]) * dmin[i];
        }
        if (total == 0) break;
        long pick = static_cast<long>(rng() % static_cast<uint64_t>(total));
        size_t i = 0;
        for (; i < set.size(); ++i) {
            pick -= static_cast<long>(dmin[i]) * dmin[i];
            if (pick < 0) break;
        }
        centers.push_back(*set[std::min(i, set.size() - 1)]);
    }
    std::vector<int> assign(set.size(), 0);
    for (int it = 0; it < 8; ++it) {
        for (size_t i = 0; i < set.size(); ++i) {
            int best = 0, bd = 1 << 30;
            for (size_t c = 0; c < centers.size(); ++c) {
                int d = hamming(*set[i], centers[c]);
                if (d < bd) {
                    bd = d;
                    best = static_cast<int>(c);
                }
            }
            assign[i] = best;
        }
        for (size_t c = 0; c < centers.size(); ++c) {
            std::vector<const Desc*> members;
            for (size_t i = 0; i < set.size(); ++i)
                if (assign[i] == static_cast<int>(c)) members.push_back(set[i]);
            if (!members.empty()) centers[c] = majority(members);
        }
    }
    for (size_t c = 0; c < centers.size(); ++c) {
        std::vector<const Desc*> members;
        for (size_t i = 0; i < set.size(); ++i)
            if (assign[i] == static_cast<int>(c)) members.push_back(set[i]);
        if (members.empty()) continue;
        int child = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{centers[c], {}, -1});
        nodes_[node].children.push_back(child);
        build(child, members, k, level + 1, levels, seed);
    }
    if (nodes_[node].children.empty()) nodes_[node].word = n_words_++;
}

void Vocabulary::train(const std::vector<Desc>& descs, int k, int levels, uint32_t seed) {
    nodes_.clear();
    n_words_ = 0;
    nodes_.push_back(Node{});
    std::vector<const Desc*> all;
    for (const Desc& d : descs) all.push_back(&d);
    build(0, all, k, 0, levels, seed);
    idf_.assign(n_words_, 1.0);
}

int Vocabulary::word(const Desc& d) const {
    int n = 0;
    while (nodes_[n].word < 0) {
        int best = -1, bd = 1 << 30;
        for (int c : nodes_[n].children) {
            int dist = hamming(d, nodes_[c].center);
            if (dist < bd) {
                bd = dist;
                best = c;
            }
        }
        if (best < 0) break;
        n = best;
    }
    return nodes_[n].word;
}

bool Vocabulary::save(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t n = static_cast<uint32_t>(nodes_.size()), w = static_cast<uint32_t>(n_words_);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(&w, 4, 1, f);
    for (const Node& nd : nodes_) {
        std::fwrite(nd.center.data(), 8, 4, f);
        int32_t word = nd.word, nc = static_cast<int32_t>(nd.children.size());
        std::fwrite(&word, 4, 1, f);
        std::fwrite(&nc, 4, 1, f);
        for (int c : nd.children) {
            int32_t cc = c;
            std::fwrite(&cc, 4, 1, f);
        }
    }
    std::fwrite(idf_.data(), 8, idf_.size(), f);
    std::fclose(f);
    return true;
}

bool Vocabulary::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0, w = 0;
    bool ok = std::fread(&n, 4, 1, f) == 1 && std::fread(&w, 4, 1, f) == 1;
    nodes_.assign(n, Node{});
    for (uint32_t i = 0; ok && i < n; ++i) {
        int32_t word = 0, nc = 0;
        ok = std::fread(nodes_[i].center.data(), 8, 4, f) == 4 && std::fread(&word, 4, 1, f) == 1 && std::fread(&nc, 4, 1, f) == 1;
        nodes_[i].word = word;
        for (int32_t c = 0; ok && c < nc; ++c) {
            int32_t cc = 0;
            ok = std::fread(&cc, 4, 1, f) == 1;
            nodes_[i].children.push_back(cc);
        }
    }
    n_words_ = static_cast<int>(w);
    idf_.assign(n_words_, 1.0);
    if (ok) ok = std::fread(idf_.data(), 8, idf_.size(), f) == idf_.size();
    std::fclose(f);
    return ok;
}

BowVector LoopDetector::bow(const std::vector<OrbFeature>& f) const {
    BowVector v;
    for (const OrbFeature& o : f) {
        int w = voc_->word(o.d);
        v[w] += voc_->idf(w);
    }
    double s = 0;
    for (auto& [w, x] : v) s += x;
    if (s > 0)
        for (auto& [w, x] : v) x /= s;
    return v;
}

double LoopDetector::score(const BowVector& a, const BowVector& b) {
    // DBoW2's L1 score: 1 - |a - b|_1 / 2 for L1-normalized vectors
    double s = 0;
    auto i = a.begin();
    auto j = b.begin();
    while (i != a.end() && j != b.end()) {
        if (i->first == j->first) {
            s += std::fabs(i->second - j->second) - i->second - j->second;
            ++i;
            ++j;
        } else if (i->first < j->first) {
            ++i;
        } else {
            ++j;
        }
    }
    return -s / 2;
}

std::vector<OrbFeature> loop_features(const Image& left, int max_features) {
    // the same histogram equalization as the front-end
    int hist[256] = {0};
    for (uint8_t v : left.px) hist[v]++;
    int lut[256], acc = 0, n = static_cast<int>(left.px.size());
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        lut[i] = static_cast<int>(255.0 * acc / n);
    }
    Image eq = left;
    for (uint8_t& v : eq.px) v = static_cast<uint8_t>(lut[v]);
    auto corners = fast9(eq, 20, 20);
    auto picked = select_spread(corners, {}, eq.w, eq.h, 10, static_cast<size_t>(max_features));
    return orb_describe(eq, picked);
}

std::vector<int> ransac_rigid(const std::vector<Vec3>& a, const std::vector<Vec3>& b, double thr, int iterations, SE3& T, uint32_t seed) {
    std::vector<int> best;
    size_t n = a.size();
    if (n < 3) return best;
    std::mt19937 rng(seed);
    for (int it = 0; it < iterations; ++it) {
        size_t i = rng() % n, j = rng() % n, k = rng() % n;
        if (i == j || j == k || i == k) continue;
        std::vector<Vec3> sa = {a[i], a[j], a[k]}, sb = {b[i], b[j], b[k]};
        // a degenerate sample: nearly collinear points
        if (cross(a[j] - a[i], a[k] - a[i]).norm() < 1e-3) continue;
        SE3 M = align_rigid(sa, sb);
        std::vector<int> in;
        for (size_t q = 0; q < n; ++q)
            if ((M * a[q] - b[q]).norm() < thr) in.push_back(static_cast<int>(q));
        if (in.size() > best.size()) best = in;
    }
    if (best.size() >= 3) {
        // refit on all inliers, then once more
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<Vec3> sa, sb;
            for (int q : best) {
                sa.push_back(a[q]);
                sb.push_back(b[q]);
            }
            T = align_rigid(sa, sb);
            std::vector<int> in;
            for (size_t q = 0; q < n; ++q)
                if ((T * a[q] - b[q]).norm() < thr) in.push_back(static_cast<int>(q));
            best = in;
        }
    }
    return best;
}

bool LoopDetector::add(LoopKeyframe kf, const SE3& T_bc, LoopConstraint& out) {
    kf.id = static_cast<int>(kfs_.size());
    kf.bow = bow(kf.orb);
    bool found = false;
    // the score of the previous keyframe: how similar consecutive views are
    double ref = kfs_.empty() ? 0.0 : score(kf.bow, kfs_.back().bow);
    // candidates: earlier keyframes sharing words, older than min_gap_s
    std::unordered_map<int, int> shared;
    for (auto& [w, x] : kf.bow) {
        auto it = inverted_.find(w);
        if (it == inverted_.end()) continue;
        for (int k : it->second)
            if (kf.t - kfs_[k].t > min_gap_s) shared[k]++;
    }
    std::vector<std::pair<double, int>> scored;
    for (auto& [k, c] : shared) {
        if (c < 10) continue;
        scored.push_back({score(kf.bow, kfs_[k].bow), k});
    }
    std::sort(scored.rbegin(), scored.rend());
    for (size_t q = 0; q < scored.size() && q < 3 && !found; ++q) {
        auto [s, k] = scored[q];
        if (s < std::max(0.05, 0.3 * ref)) break;
        ++candidates_checked;
        const LoopKeyframe& old = kfs_[k];
        // descriptor matches between points with depth on both sides, with a ratio test
        std::vector<Vec3> pa, pb;
        for (size_t i = 0; i < kf.orb.size(); ++i) {
            if (kf.pts[i][2] <= 0) continue;
            int b1 = 256, b2 = 256, bj = -1;
            for (size_t j = 0; j < old.orb.size(); ++j) {
                if (old.pts[j][2] <= 0) continue;
                int d = hamming(kf.orb[i].d, old.orb[j].d);
                if (d < b1) {
                    b2 = b1;
                    b1 = d;
                    bj = static_cast<int>(j);
                } else if (d < b2) {
                    b2 = d;
                }
            }
            if (bj < 0 || b1 > 60 || b1 > 0.8 * b2) continue;
            pa.push_back(kf.pts[i]);
            pb.push_back(old.pts[bj]);
        }
        static const bool dbg = std::getenv("ROVER_LOOPDBG") != nullptr;
        if (dbg) {
            int with_depth = 0;
            for (auto& p : kf.pts) with_depth += p[2] > 0;
            std::fprintf(stderr, "loop? t %.1f old %.1f score %.3f ref %.3f features %zu depth %d matches %zu\n", kf.t, old.t, s, ref, kf.orb.size(), with_depth, pa.size());
        }
        if (static_cast<int>(pa.size()) < min_inliers) continue;
        SE3 T_oc;  // current left camera -> old left camera
        std::vector<int> in = ransac_rigid(pa, pb, ransac_threshold, 300, T_oc);
        if (dbg) std::fprintf(stderr, "   inliers %zu\n", in.size());
        if (static_cast<int>(in.size()) < min_inliers) continue;
        ++verified;
        out.i = k;
        out.j = kf.id;
        out.T_ij = T_bc * T_oc * T_bc.inverse();  // body of current in the body frame of old
        out.inliers = static_cast<int>(in.size());
        out.score = s;
        found = true;
    }
    for (auto& [w, x] : kf.bow) inverted_[w].push_back(kf.id);
    kfs_.push_back(std::move(kf));
    return found;
}

void PoseGraph::add_keyframe(double t, const SE3& T_odom) {
    t_.push_back(t);
    odom_.push_back(T_odom);
    // until optimized, a new keyframe moves with the last correction
    opt_.push_back(opt_.empty() ? T_odom : correction(opt_.size() - 1) * T_odom);
}

void PoseGraph::add_loop(const LoopConstraint& c) { loops_.push_back(c); }

SolveStats PoseGraph::optimize() {
    Problem p;
    std::vector<Var*> v;
    for (size_t k = 0; k < odom_.size(); ++k) v.push_back(p.add(Var::pose4(opt_[k])));
    v[0]->fixed = true;
    for (size_t k = 0; k + 1 < odom_.size(); ++k) {
        // the 4-DoF odometry edge carries the relative motion of the odometry
        auto f = std::make_unique<RelPose4Factor>(v[k], v[k + 1], odom_[k].inverse() * odom_[k + 1], 0.01, 0.02);
        p.add_factor(std::move(f));
    }
    for (const LoopConstraint& c : loops_) {
        auto f = std::make_unique<RelPose4Factor>(v[c.i], v[c.j], c.T_ij, 0.01, 0.02);
        f->huber = 3.0;
        p.add_factor(std::move(f));
    }
    SolveStats st = p.solve(30);
    for (size_t k = 0; k < odom_.size(); ++k) opt_[k] = v[k]->pose_value();
    return st;
}

}  // namespace rover
