#include "rover/frontend.hpp"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rover {

Image equalize(const Image& img) {
    int hist[256] = {0};
    for (uint8_t v : img.px) hist[v]++;
    int lut[256], acc = 0, n = static_cast<int>(img.px.size());
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        lut[i] = static_cast<int>(255.0 * acc / n);
    }
    Image out = img;
    for (uint8_t& v : out.px) v = static_cast<uint8_t>(lut[v]);
    return out;
}

double triangulate_depth(const Vec2& xl, const Vec2& xr, const SE3& T_rl) {
    // X_r = R (d * fl) + t, with fl = (xl, 1); solve for d minimizing the
    // distance to the ray of xr: two linear equations in d
    Vec3 fl{xl[0], xl[1], 1.0};
    Vec3 a = T_rl.R * fl, t = T_rl.t;
    // xr * (a_z d + t_z) = a_x d + t_x ; same for y
    double A0 = a[0] - xr[0] * a[2], b0 = xr[0] * t[2] - t[0];
    double A1 = a[1] - xr[1] * a[2], b1 = xr[1] * t[2] - t[1];
    double den = A0 * A0 + A1 * A1;
    if (den < 1e-12) return -1;
    return (A0 * b0 + A1 * b1) / den;
}

Frontend::Frontend(const Camera& left, const Camera& right, const FrontendConfig& cfg) : cfg_(cfg) {
    cam_[0] = left;
    cam_[1] = right;
    T_c1c0_ = right.T_bc.inverse() * left.T_bc;
    E_ = skew(T_c1c0_.t) * T_c1c0_.R;
}

void Frontend::reject(const std::vector<int>& ids) {
    if (ids.empty()) return;
    std::vector<int> keep_ids;
    std::vector<Vec2> keep_pts;
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (std::find(ids.begin(), ids.end(), ids_[i]) != ids.end()) {
            obs_.erase(ids_[i]);
            continue;
        }
        keep_ids.push_back(ids_[i]);
        keep_pts.push_back(pts_[i]);
    }
    ids_.swap(keep_ids);
    pts_.swap(keep_pts);
}

const std::unordered_map<int, Observation>& Frontend::process(const Image& left_in, const Image& right_in) {
    Image left = cfg_.equalize ? equalize(left_in) : left_in;
    Image right = cfg_.equalize ? equalize(right_in) : right_in;
    std::vector<ImageF> L = pyramid(left, 4), R = pyramid(right, 4);
    if (std::getenv("ROVER_KLT21")) klt_.half_window = 10;
    obs_.clear();
    tracked_ = created_ = 0;
    // 1. track the existing features from the previous left image
    if (!prev_left_.empty() && !pts_.empty()) {
        std::vector<Vec2> fwd = pts_, back;
        std::vector<uint8_t> ok, ok2;
        klt_track(prev_left_, L, pts_, fwd, ok, klt_);
        back = pts_;
        klt_track(L, prev_left_, fwd, back, ok2, klt_);
        std::vector<int> ids;
        std::vector<Vec2> pts;
        for (size_t i = 0; i < pts_.size(); ++i) {
            if (!ok[i] || !ok2[i]) continue;
            if ((back[i] - pts_[i]).norm() > cfg_.max_fb_error) continue;
            if (!cam_[0].inside(fwd[i], 4)) continue;
            ids.push_back(ids_[i]);
            pts.push_back(fwd[i]);
        }
        ids_.swap(ids);
        pts_.swap(pts);
        tracked_ = static_cast<int>(pts_.size());
    } else {
        ids_.clear();
        pts_.clear();
    }
    // 2. new corners where there are few
    if (static_cast<int>(pts_.size()) < cfg_.max_features) {
        auto corners = fast9(left, cfg_.fast_threshold, 12);
        auto fresh = select_spread(corners, pts_, left.w, left.h, cfg_.min_distance, cfg_.max_features - pts_.size());
        for (const Corner& c : fresh) {
            ids_.push_back(next_id_++);
            pts_.push_back(Vec2{c.x, c.y});
            ++created_;
        }
    }
    for (size_t i = 0; i < pts_.size(); ++i) {
        Observation o;
        o.uv[0] = pts_[i];
        o.xn[0] = cam_[0].unproject(pts_[i]);
        obs_[ids_[i]] = o;
    }
    // 3. the right image
    stereo_match(L, R);
    prev_left_ = std::move(L);
    return obs_;
}

std::vector<double> Frontend::stereo_depths(const Image& left_in, const Image& right_in, const std::vector<Vec2>& pts) const {
    Image left = cfg_.equalize ? equalize(left_in) : left_in;
    Image right = cfg_.equalize ? equalize(right_in) : right_in;
    std::vector<ImageF> L = pyramid(left, 4), R = pyramid(right, 4);
    std::vector<Vec2> guess = pts, back;
    std::vector<uint8_t> ok, ok2;
    klt_track(L, R, pts, guess, ok, klt_);
    back = pts;
    klt_track(R, L, guess, back, ok2, klt_);
    std::vector<double> depth(pts.size(), -1.0);
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!ok[i] || !ok2[i] || (back[i] - pts[i]).norm() > cfg_.max_fb_error) continue;
        Vec2 xl = cam_[0].unproject(pts[i]), xr;
        double residual, dy;
        if (!epipolar(xl, guess[i], xr, residual, dy) || std::fabs(residual) > cfg_.max_epipolar) continue;
        double d = triangulate_depth(xl, xr, T_c1c0_);
        if (d > 0.2 && d < 20) depth[i] = d;
    }
    return depth;
}

bool Frontend::epipolar(const Vec2& xl, const Vec2& right_px, Vec2& xr, double& residual, double& dy) const {
    Vec2 corrected = right_px;
    corrected[1] += stereo_dy_;
    xr = cam_[1].unproject(corrected);
    Vec3 xl3{xl[0], xl[1], 1.0}, xr3{xr[0], xr[1], 1.0};
    Vec3 l = E_ * xl3;  // the epipolar line of xl in the right image
    double n = std::sqrt(l[0] * l[0] + l[1] * l[1]);
    if (!(n > 0)) return false;
    residual = dot(xr3, l) / n * cam_[1].fx;  // signed distance, pixels
    double slope = l[1] / n * cam_[1].fx / cam_[1].fy;  // d residual / d row
    dy = std::fabs(slope) > 0.5 ? -residual / slope : 0.0;
    return true;
}

void Frontend::stereo_match(const std::vector<ImageF>& L, const std::vector<ImageF>& R) {
    if (pts_.empty()) return;
    std::vector<Vec2> guess = pts_, back;
    std::vector<uint8_t> ok, ok2;
    klt_track(L, R, pts_, guess, ok, klt_);
    back = pts_;
    klt_track(R, L, guess, back, ok2, klt_);
    std::vector<double> shifts;  // the row shift each match would need to lie on its epipolar line
    for (size_t i = 0; i < pts_.size(); ++i) {
        if (!ok[i] || !ok2[i] || (back[i] - pts_[i]).norm() > cfg_.max_fb_error) continue;
        if (!cam_[1].inside(guess[i], 4)) continue;
        Observation& o = obs_[ids_[i]];
        Vec2 xr;
        double residual, dy;
        if (!epipolar(o.xn[0], guess[i], xr, residual, dy)) continue;
        if (std::fabs(residual) < 6 && dy != 0) shifts.push_back(dy);
        if (std::fabs(residual) > cfg_.max_epipolar) continue;
        double d = triangulate_depth(o.xn[0], xr, T_c1c0_);
        if (!(d > 0.1 && d < 40)) continue;
        o.uv[1] = guess[i];
        o.xn[1] = xr;
        o.stereo = true;
    }
    // Self-calibration of the stereo rig's vertical misalignment: a small
    // rotation error about the baseline moves every right-image point by about
    // the same number of rows. The median shift of this frame's matches
    // (KLT does not use the calibration) corrects it, quickly at first and
    // then slowly.
    if (cfg_.self_calibrate && shifts.size() >= 15) {
        std::nth_element(shifts.begin(), shifts.begin() + shifts.size() / 2, shifts.end());
        double gain = calib_updates_ < 20 ? 0.5 : 0.05;
        stereo_dy_ = std::clamp(stereo_dy_ + gain * shifts[shifts.size() / 2], -5.0, 5.0);
        ++calib_updates_;
    }
}

}  // namespace rover
