// Loop closure: a vocabulary of binary visual words (hierarchical
// k-majority clustering of ORB descriptors, Galvez-Lopez and Tardos 2012),
// tf-idf bag-of-words vectors with an inverted index to find earlier
// keyframes that look alike, a geometric check with 3D-3D RANSAC between the
// stereo points of the two keyframes, and a 4-DoF pose graph over all
// keyframes that spreads the correction.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "features.hpp"
#include "lsq.hpp"
#include "math.hpp"

namespace rover {

using Desc = std::array<uint64_t, 4>;

class Vocabulary {
public:
    /// k branches, `levels` deep (k^levels words at most)
    void train(const std::vector<Desc>& descs, int k = 10, int levels = 4, uint32_t seed = 1);
    int word(const Desc& d) const;
    int words() const { return n_words_; }
    void set_idf(const std::vector<double>& idf) { idf_ = idf; }
    double idf(int w) const { return w < static_cast<int>(idf_.size()) ? idf_[w] : 1.0; }
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    struct Node {
        Desc center{};
        std::vector<int> children;
        int word = -1;
    };
    std::vector<Node> nodes_;
    std::vector<double> idf_;
    int n_words_ = 0;
    void build(int node, std::vector<const Desc*>& set, int k, int level, int levels, uint32_t& seed);
};

using BowVector = std::map<int, double>;  // word -> weight (L1-normalized)

struct LoopKeyframe {
    int id = 0;
    double t = 0;
    SE3 T_wb;                     // the odometry pose when it was stored
    std::vector<OrbFeature> orb;  // left image
    std::vector<Vec3> pts;        // the 3D point (left camera) of each feature, z <= 0 if none
    BowVector bow;
};

struct LoopConstraint {
    int i, j;     // j: the current keyframe, i: the earlier one
    SE3 T_ij;     // body of j in the body frame of i
    int inliers;
    double score;
};

class LoopDetector {
public:
    explicit LoopDetector(const Vocabulary* voc) : voc_(voc) {}
    BowVector bow(const std::vector<OrbFeature>& f) const;
    /// add a keyframe; returns a verified loop with an earlier keyframe, if any
    bool add(LoopKeyframe kf, const SE3& T_bc, LoopConstraint& out);
    const std::vector<LoopKeyframe>& keyframes() const { return kfs_; }
    double min_gap_s = 20;      // only keyframes at least this much older
    int min_inliers = 30;
    double ransac_threshold = 0.08;  // meters
    int candidates_checked = 0, verified = 0;

private:
    const Vocabulary* voc_;
    std::vector<LoopKeyframe> kfs_;
    std::unordered_map<int, std::vector<int>> inverted_;  // word -> keyframes
    static double score(const BowVector& a, const BowVector& b);
};

/// ORB features of a left image (FAST corners spread over the image) for
/// place recognition.
std::vector<OrbFeature> loop_features(const Image& left, int max_features = 600);

/// 3D-3D RANSAC: the rigid motion T with b ~ T a; returns the inliers.
std::vector<int> ransac_rigid(const std::vector<Vec3>& a, const std::vector<Vec3>& b, double threshold, int iterations, SE3& T, uint32_t seed = 7);

/// A 4-DoF pose graph over keyframes: odometry edges between consecutive
/// keyframes and loop edges.
class PoseGraph {
public:
    void add_keyframe(double t, const SE3& T_odom);
    void add_loop(const LoopConstraint& c);
    /// optimize; afterwards corrected(k) is the optimized pose of keyframe k
    SolveStats optimize();
    size_t size() const { return odom_.size(); }
    SE3 corrected(size_t k) const { return opt_[k]; }
    SE3 odom(size_t k) const { return odom_[k]; }
    double time(size_t k) const { return t_[k]; }
    const std::vector<LoopConstraint>& loops() const { return loops_; }
    /// the correction to apply to an odometry pose taken after keyframe k
    SE3 correction(size_t k) const { return opt_[k] * odom_[k].inverse(); }

private:
    std::vector<double> t_;
    std::vector<SE3> odom_, opt_;
    std::vector<LoopConstraint> loops_;
};

}  // namespace rover
