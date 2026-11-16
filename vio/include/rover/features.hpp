// Image features: FAST corners spread over the image, pyramidal
// Lucas-Kanade tracking, and ORB descriptors for place recognition.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"
#include "math.hpp"

namespace rover {

struct Corner {
    float x, y;
    int score;
};

/// FAST-9 corners (16-pixel circle of radius 3, 9 contiguous pixels all
/// brighter or all darker than the center by `threshold`), with 3x3
/// non-maximum suppression on a sum-of-differences score.
std::vector<Corner> fast9(const Image& img, int threshold, int border = 8);

/// Pick corners by decreasing score, at least min_dist apart from each
/// other and from the existing points.
std::vector<Corner> select_spread(std::vector<Corner> corners, const std::vector<Vec2>& existing, int w, int h, double min_dist, size_t max_new);

struct KltParams {
    int half_window = 7;  // 15 x 15
    int levels = 4;
    int iterations = 20;
    double epsilon = 0.01;  // pixels
};

/// Track points from image a to image b (pyramids). `guess` holds initial
/// positions in b (often the same as in a); `ok` tells which converged.
void klt_track(const std::vector<ImageF>& a, const std::vector<ImageF>& b, const std::vector<Vec2>& pts, std::vector<Vec2>& guess, std::vector<uint8_t>& ok,
               const KltParams& p = {});

/// ORB: oriented FAST keypoint with a rotated BRIEF descriptor (256 bits).
struct OrbFeature {
    float x, y, angle;
    std::array<uint64_t, 4> d;
};
std::vector<OrbFeature> orb_describe(const Image& img, const std::vector<Corner>& corners);
inline int hamming(const std::array<uint64_t, 4>& a, const std::array<uint64_t, 4>& b) {
    return __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]) + __builtin_popcountll(a[2] ^ b[2]) + __builtin_popcountll(a[3] ^ b[3]);
}

}  // namespace rover
