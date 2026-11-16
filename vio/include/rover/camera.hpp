// A pinhole camera with radial-tangential distortion (the EuRoC model), and
// the extrinsics of each camera relative to the body (IMU) frame.
#pragma once

#include "math.hpp"

namespace rover {

struct Camera {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0;
    int width = 0, height = 0;
    SE3 T_bc;  // camera to body: x_b = T_bc x_c

    /// distort a point on the normalized image plane
    Vec2 distort(double x, double y) const {
        double r2 = x * x + y * y, rad = 1 + k1 * r2 + k2 * r2 * r2;
        return Vec2{x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y};
    }
    /// camera-frame point to pixel
    bool project(const Vec3& Xc, Vec2& uv) const {
        if (Xc[2] <= 1e-6) return false;
        Vec2 d = distort(Xc[0] / Xc[2], Xc[1] / Xc[2]);
        uv = Vec2{fx * d[0] + cx, fy * d[1] + cy};
        return true;
    }
    /// pixel to the normalized image plane (undistorted), by fixed-point
    /// iteration refined with Gauss-Newton
    Vec2 unproject(const Vec2& uv) const {
        double xd = (uv[0] - cx) / fx, yd = (uv[1] - cy) / fy;
        double x = xd, y = yd;
        for (int it = 0; it < 20; ++it) {
            Vec2 d = distort(x, y);
            double ex = d[0] - xd, ey = d[1] - yd;
            if (ex * ex + ey * ey < 1e-24) break;
            // Jacobian of distort
            double h = 1e-7;
            Vec2 dx = distort(x + h, y), dy = distort(x, y + h);
            double a = (dx[0] - d[0]) / h, b = (dy[0] - d[0]) / h, c = (dx[1] - d[1]) / h, e = (dy[1] - d[1]) / h;
            double det = a * e - b * c;
            x -= (e * ex - b * ey) / det;
            y -= (-c * ex + a * ey) / det;
        }
        return Vec2{x, y};
    }
    Vec3 bearing(const Vec2& uv) const {
        Vec2 n = unproject(uv);
        Vec3 v{n[0], n[1], 1.0};
        return v * (1.0 / v.norm());
    }
    bool inside(const Vec2& uv, double margin = 0) const {
        return uv[0] >= margin && uv[1] >= margin && uv[0] < width - 1 - margin && uv[1] < height - 1 - margin;
    }
};

struct ImuNoise {
    double gyro_noise = 1.6968e-4;   // rad / s / sqrt(Hz)
    double gyro_walk = 1.9393e-5;    // rad / s^2 / sqrt(Hz)
    double acc_noise = 2.0e-3;       // m / s^2 / sqrt(Hz)
    double acc_walk = 3.0e-3;        // m / s^3 / sqrt(Hz)
};

}  // namespace rover
