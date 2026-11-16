// The EuRoC MAV dataset in ASL format: mav0/cam0, cam1 (752x480 grayscale
// PNG at 20 Hz), imu0 (200 Hz), state_groundtruth_estimate0 (Vicon or Leica
// with the IMU biases), and each sensor's sensor.yaml.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "camera.hpp"
#include "math.hpp"

namespace rover {

struct ImuSample {
    double t;  // seconds
    Vec3 gyro, acc;
};

struct GroundTruth {
    double t;
    SE3 T_wb;
    Vec3 v, bg, ba;
};

struct Frame {
    double t;
    std::string left, right;  // image paths
};

struct EurocSequence {
    std::string root;  // the directory holding mav0
    Camera cam[2];
    ImuNoise noise;
    std::vector<ImuSample> imu;
    std::vector<Frame> frames;
    std::vector<GroundTruth> truth;
    int64_t t0_ns = 0;  // all times are seconds after this

    /// Load everything but the images. Throws on missing files.
    static EurocSequence load(const std::string& root);
    /// Load only the cameras and the IMU noise (the sensor.yaml files).
    static EurocSequence load_calibration(const std::string& root);
    /// Ground truth at time t (linear interpolation of position, slerp of rotation); false outside.
    bool truth_at(double t, SE3& T) const;
};

/// Parse "key: [a, b, ...]" numbers (and "data: [...]" of a matrix) from a sensor.yaml.
std::vector<double> yaml_numbers(const std::string& text, const std::string& key);

}  // namespace rover
