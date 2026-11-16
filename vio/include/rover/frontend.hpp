// The visual front-end: corners tracked from frame to frame in the left
// camera (KLT with a forward-backward check), matched into the right camera
// (KLT along the baseline, checked with the epipolar geometry), and new
// corners added where the image has few.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "camera.hpp"
#include "features.hpp"
#include "image.hpp"

namespace rover {

struct Observation {
    Vec2 uv[2];        // pixels in the left and right image
    Vec2 xn[2];        // the same on the normalized (undistorted) image plane
    bool stereo = false;
};

struct FrontendConfig {
    int max_features = 180;
    double min_distance = 25;  // pixels between features
    int fast_threshold = 15;
    double max_fb_error = 0.7;  // pixels, forward-backward
    double max_epipolar = 1.5;  // pixels
    bool equalize = true;
    bool self_calibrate = true;  // estimate the rig's vertical misalignment online
};

class Frontend {
public:
    Frontend(const Camera& left, const Camera& right, const FrontendConfig& cfg = {});

    /// Process a stereo pair; returns the observations of this frame by feature id.
    const std::unordered_map<int, Observation>& process(const Image& left, const Image& right);

    /// Drop features (e.g. outliers found by the back-end).
    void reject(const std::vector<int>& ids);

    /// Depth (left camera) of arbitrary left-image points, matched into the
    /// right image like the tracked features; <= 0 where there is no match.
    std::vector<double> stereo_depths(const Image& left, const Image& right, const std::vector<Vec2>& pts) const;

    int tracked_last() const { return tracked_; }
    int created_last() const { return created_; }
    const std::vector<ImageF>& left_pyramid() const { return prev_left_; }
    /// The estimated row offset of the right image (pixels) from self-calibration.
    double stereo_row_offset() const { return stereo_dy_; }

private:
    Camera cam_[2];
    FrontendConfig cfg_;
    KltParams klt_;
    SE3 T_c1c0_;  // left camera frame to right camera frame
    Mat3 E_;      // essential matrix for the epipolar check
    std::vector<ImageF> prev_left_;
    std::vector<int> ids_;
    std::vector<Vec2> pts_;
    std::unordered_map<int, Observation> obs_;
    int next_id_ = 0, tracked_ = 0, created_ = 0;
    double stereo_dy_ = 0;  // row correction added to right-image points
    int calib_updates_ = 0;

    /// The right point corrected and unprojected (xr), its signed epipolar
    /// residual (pixels) and the row shift that would cancel it.
    bool epipolar(const Vec2& xl, const Vec2& right_px, Vec2& xr, double& residual, double& dy) const;
    void stereo_match(const std::vector<ImageF>& L, const std::vector<ImageF>& R);
};

/// Equalize the histogram of an 8-bit image (EuRoC has strong exposure changes).
Image equalize(const Image& img);

/// Triangulate a point seen in the left and right camera (normalized
/// coordinates); returns the depth in the left camera, or a negative value.
double triangulate_depth(const Vec2& xl, const Vec2& xr, const SE3& T_rl);

}  // namespace rover
