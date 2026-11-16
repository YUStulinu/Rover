// Stereo visual-inertial odometry: a sliding window of frames (pose,
// velocity, IMU biases) and landmarks (inverse depth in the left camera of
// the frame that first triangulated them), optimized together with IMU
// preintegration factors, reprojection factors in both cameras and the
// prior left by marginalizing old frames.
//
// Position and yaw are not observable: a prior on the first frame's pose
// sets them, and its information is carried by the marginalization prior
// from then on. When the window is full, the second newest frame is dropped
// if it adds little parallax (its IMU data is merged into the next
// interval); otherwise the oldest frame is marginalized: its pose,
// velocity, biases and landmarks are removed by a Schur complement, which
// leaves a linear prior on the frames they were connected to.
#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "camera.hpp"
#include "dense.hpp"
#include "frontend.hpp"
#include "imu.hpp"
#include "lsq.hpp"

namespace rover {

struct EstimatorConfig {
    int window = 10;
    double pixel_sigma = 1.0;
    double huber_px = 1.5;
    double keyframe_parallax_px = 12.0;
    int keyframe_min_common = 60;
    int iterations = 6;
    double max_reprojection_px = 4.0;
    double min_depth = 0.1, max_depth = 40;
};

struct FrameState {
    double t = 0;
    std::unique_ptr<Var> pose, vel, bias;  // bias: gyro (3) then accelerometer (3)
    Preintegration pre;                    // from the previous frame in the window
    std::unordered_map<int, Observation> obs;
    bool keyframe = true;
    int index = 0;  // a running number, for output
};

struct Landmark {
    FrameState* anchor = nullptr;
    Vec2 xn;  // normalized coordinates in the anchor's left camera
    std::unique_ptr<Var> rho;  // inverse depth
};

/// The prior left by marginalization: r(x) = r0 + J (x (-) x0) on the kept variables.
struct MarginalPrior : Factor {
    std::vector<std::array<double, 16>> x0;
    MatX J;
    std::vector<double> r0;
    void residual(double* r) const override;
};

class Estimator {
public:
    Estimator(const Camera& left, const Camera& right, const ImuNoise& noise, const EstimatorConfig& cfg = {});

    /// Add a frame: the IMU samples since the previous frame (each with the
    /// interval it covers up to the next sample) and the front-end's
    /// observations. Returns false until initialized.
    bool add_frame(double t, const std::vector<ImuSample>& imu, const std::unordered_map<int, Observation>& obs);

    SE3 pose() const { return frames_.back()->pose->pose_value(); }
    bool last_is_keyframe() const { return !frames_.empty() && frames_.back()->keyframe; }
    Vec3 velocity() const { return frames_.back()->vel->vec3(); }
    Vec3 gyro_bias() const { return frames_.back()->bias->vec3(0); }
    Vec3 acc_bias() const { return frames_.back()->bias->vec3(3); }
    /// features the back-end found to be outliers in the last frame
    const std::vector<int>& outliers() const { return outliers_; }
    size_t landmarks() const { return lm_.size(); }
    /// world points of the current landmarks (for display)
    std::vector<Vec3> landmark_points() const;
    /// the frames in the window: (time, pose)
    std::vector<std::pair<double, SE3>> window_poses() const;
    double last_cost = 0;
    int last_iterations = 0;

private:
    Camera cam_[2];
    ImuNoise noise_;
    EstimatorConfig cfg_;
    std::deque<std::unique_ptr<FrameState>> frames_;
    std::unordered_map<int, Landmark> lm_;
    std::unique_ptr<MarginalPrior> prior_;
    std::vector<int> outliers_;
    bool initialized_ = false;
    bool initial_prior_ = false;
    SE3 initial_pose_;
    int frame_count_ = 0;

    void build(Problem& p, bool with_prior);
    void optimize();
    void remove_outliers();
    void slide();
    void drop_second_newest();
    void marginalize_oldest();
    void reanchor(int id, Landmark& l);
    Vec3 landmark_world(const Landmark& l) const;
};

}  // namespace rover
