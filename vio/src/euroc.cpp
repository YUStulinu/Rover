#include "rover/euroc.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rover {

static std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<double> yaml_numbers(const std::string& text, const std::string& key) {
    size_t k = text.find(key + ":");
    if (k == std::string::npos) return {};
    size_t a = text.find('[', k), b = text.find(']', a);
    if (a == std::string::npos || b == std::string::npos) {
        // a single number after the key
        size_t s = k + key.size() + 1;
        return {std::strtod(text.c_str() + s, nullptr)};
    }
    std::vector<double> v;
    std::string body = text.substr(a + 1, b - a - 1);
    for (char& c : body)
        if (c == ',') c = ' ';
    std::stringstream ss(body);
    double x;
    while (ss >> x) v.push_back(x);
    return v;
}

static std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) f.push_back(cell);
        rows.push_back(f);
    }
    return rows;
}

static Camera load_camera(const std::string& yaml_path) {
    std::string y = read_text(yaml_path);
    Camera c;
    auto in = yaml_numbers(y, "intrinsics");
    auto d = yaml_numbers(y, "distortion_coefficients");
    auto res = yaml_numbers(y, "resolution");
    auto T = yaml_numbers(y.substr(y.find("T_BS")), "data");
    if (in.size() != 4 || d.size() != 4 || res.size() != 2 || T.size() != 16) throw std::runtime_error("bad camera yaml " + yaml_path);
    c.fx = in[0];
    c.fy = in[1];
    c.cx = in[2];
    c.cy = in[3];
    c.k1 = d[0];
    c.k2 = d[1];
    c.p1 = d[2];
    c.p2 = d[3];
    c.width = static_cast<int>(res[0]);
    c.height = static_cast<int>(res[1]);
    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 3; ++k) c.T_bc.R(r, k) = T[r * 4 + k];
        c.T_bc.t[r] = T[r * 4 + 3];
    }
    c.T_bc.R = orthonormalize(c.T_bc.R);
    return c;
}

EurocSequence EurocSequence::load_calibration(const std::string& root) {
    EurocSequence s;
    s.root = root;
    std::string mav = root + "/mav0/";
    s.cam[0] = load_camera(mav + "cam0/sensor.yaml");
    s.cam[1] = load_camera(mav + "cam1/sensor.yaml");
    std::string iy = read_text(mav + "imu0/sensor.yaml");
    auto num = [&](const char* k, double d) {
        auto v = yaml_numbers(iy, k);
        return v.empty() ? d : v[0];
    };
    s.noise.gyro_noise = num("gyroscope_noise_density", s.noise.gyro_noise);
    s.noise.gyro_walk = num("gyroscope_random_walk", s.noise.gyro_walk);
    s.noise.acc_noise = num("accelerometer_noise_density", s.noise.acc_noise);
    s.noise.acc_walk = num("accelerometer_random_walk", s.noise.acc_walk);
    return s;
}

EurocSequence EurocSequence::load(const std::string& root) {
    EurocSequence s;
    s.root = root;
    std::string mav = root + "/mav0/";
    s.cam[0] = load_camera(mav + "cam0/sensor.yaml");
    s.cam[1] = load_camera(mav + "cam1/sensor.yaml");
    std::string iy = read_text(mav + "imu0/sensor.yaml");
    auto num = [&](const char* k, double d) {
        auto v = yaml_numbers(iy, k);
        return v.empty() ? d : v[0];
    };
    s.noise.gyro_noise = num("gyroscope_noise_density", s.noise.gyro_noise);
    s.noise.gyro_walk = num("gyroscope_random_walk", s.noise.gyro_walk);
    s.noise.acc_noise = num("accelerometer_noise_density", s.noise.acc_noise);
    s.noise.acc_walk = num("accelerometer_random_walk", s.noise.acc_walk);

    auto imu = read_csv(mav + "imu0/data.csv");
    auto c0 = read_csv(mav + "cam0/data.csv");
    auto c1 = read_csv(mav + "cam1/data.csv");
    if (imu.empty() || c0.empty()) throw std::runtime_error("empty sequence " + root);
    s.t0_ns = std::min(std::strtoll(imu[0][0].c_str(), nullptr, 10), std::strtoll(c0[0][0].c_str(), nullptr, 10));
    auto sec = [&](const std::string& ns) { return (std::strtoll(ns.c_str(), nullptr, 10) - s.t0_ns) * 1e-9; };
    for (auto& r : imu)
        s.imu.push_back({sec(r[0]), Vec3{std::stod(r[1]), std::stod(r[2]), std::stod(r[3])}, Vec3{std::stod(r[4]), std::stod(r[5]), std::stod(r[6])}});
    // stereo pairs with the same timestamp
    std::vector<std::pair<std::string, std::string>> right;
    for (auto& r : c1) right.push_back({r[0], r[1]});
    std::sort(right.begin(), right.end());
    for (auto& r : c0) {
        auto it = std::lower_bound(right.begin(), right.end(), std::make_pair(r[0], std::string()));
        if (it == right.end() || it->first != r[0]) continue;
        std::string name = r[1];
        while (!name.empty() && (name.back() == ' ' || name.back() == '\r')) name.pop_back();
        std::string rn = it->second;
        while (!rn.empty() && (rn.back() == ' ' || rn.back() == '\r')) rn.pop_back();
        s.frames.push_back({sec(r[0]), mav + "cam0/data/" + name, mav + "cam1/data/" + rn});
    }
    std::ifstream gt_in(mav + "state_groundtruth_estimate0/data.csv");
    if (gt_in) {
        for (auto& r : read_csv(mav + "state_groundtruth_estimate0/data.csv")) {
            if (r.size() < 17) continue;
            GroundTruth g;
            g.t = sec(r[0]);
            g.T_wb.t = Vec3{std::stod(r[1]), std::stod(r[2]), std::stod(r[3])};
            g.T_wb.R = Quat(std::stod(r[4]), std::stod(r[5]), std::stod(r[6]), std::stod(r[7])).normalized().matrix();
            g.v = Vec3{std::stod(r[8]), std::stod(r[9]), std::stod(r[10])};
            g.bg = Vec3{std::stod(r[11]), std::stod(r[12]), std::stod(r[13])};
            g.ba = Vec3{std::stod(r[14]), std::stod(r[15]), std::stod(r[16])};
            s.truth.push_back(g);
        }
    }
    return s;
}

bool EurocSequence::truth_at(double t, SE3& T) const {
    if (truth.empty() || t < truth.front().t || t > truth.back().t) return false;
    auto it = std::lower_bound(truth.begin(), truth.end(), t, [](const GroundTruth& g, double x) { return g.t < x; });
    if (it == truth.begin()) {
        T = it->T_wb;
        return true;
    }
    const GroundTruth& b = *it;
    const GroundTruth& a = *(it - 1);
    if (b.t - a.t > 0.1) return false;  // a gap in the ground truth
    double u = (t - a.t) / (b.t - a.t);
    Vec3 dr = so3_log(a.T_wb.R.transpose() * b.T_wb.R);
    T.R = a.T_wb.R * so3_exp(u * dr);
    T.t = a.T_wb.t + u * (b.T_wb.t - a.T_wb.t);
    return true;
}

}  // namespace rover
