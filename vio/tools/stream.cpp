// rover-stream: the odometry as a process that reads sensor data on stdin
// and answers with poses, for closed-loop control.
//
//   rover-stream CALIBRATION_DIR   (a directory with mav0/cam0, cam1, imu0 sensor.yaml)
//
// Input lines:
//   imu T GX GY GZ AX AY AZ        (seconds; rad/s; m/s^2)
//   frame T LEFT.png RIGHT.png
// After each frame it prints:
//   pose T X Y Z QX QY QZ QW       (the body in the odometry frame; "init" before it starts)
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "rover/estimator.hpp"
#include "rover/euroc.hpp"
#include "rover/frontend.hpp"

using namespace rover;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: rover-stream CALIBRATION_DIR\n");
        return 2;
    }
    EurocSequence cal = EurocSequence::load_calibration(argv[1]);
    Frontend fe(cal.cam[0], cal.cam[1]);
    Estimator est(cal.cam[0], cal.cam[1], cal.noise);
    std::vector<ImuSample> buffer;  // raw samples not yet used
    double t_prev = -1;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        if (kind == "imu") {
            ImuSample s;
            in >> s.t >> s.gyro[0] >> s.gyro[1] >> s.gyro[2] >> s.acc[0] >> s.acc[1] >> s.acc[2];
            buffer.push_back(s);
        } else if (kind == "frame") {
            double t;
            std::string l, r;
            in >> t >> l >> r;
            std::vector<ImuSample> imu;
            // the last reading holds until the frame time
            if (!buffer.empty() && buffer.back().t < t) {
                ImuSample last = buffer.back();
                last.t = t;
                buffer.push_back(last);
            }
            if (t_prev < 0) {
                for (auto& s : buffer)
                    if (std::fabs(s.t - t) < 0.25) imu.push_back(s);
            } else {
                for (size_t i = 0; i + 1 < buffer.size(); ++i) {
                    double s0 = std::max(buffer[i].t, t_prev), s1 = std::min(buffer[i + 1].t, t);
                    if (s1 <= s0) continue;
                    imu.push_back({s1 - s0, 0.5 * (buffer[i].gyro + buffer[i + 1].gyro), 0.5 * (buffer[i].acc + buffer[i + 1].acc)});
                }
            }
            const auto& obs = fe.process(read_png(l), read_png(r));
            bool ok = est.add_frame(t, imu, obs);
            fe.reject(est.outliers());
            if (ok) {
                t_prev = t;
                // keep the last sample: it starts the next interval
                if (!buffer.empty()) buffer.erase(buffer.begin(), buffer.end() - 1);
                SE3 T = est.pose();
                Quat q = Quat::from_matrix(T.R);
                std::printf("pose %.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", t, T.t[0], T.t[1], T.t[2], q.x, q.y, q.z, q.w);
            } else {
                std::printf("init %.6f\n", t);
            }
            std::fflush(stdout);
        } else if (kind == "quit") {
            break;
        }
    }
    return 0;
}
