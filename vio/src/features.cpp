#include "rover/features.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace rover {

static const int kCircle[16][2] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
                                   {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

std::vector<Corner> fast9(const Image& img, int t, int border) {
    int w = img.w, h = img.h;
    border = std::max(border, 3);
    std::vector<int> score(static_cast<size_t>(w) * h, 0);
    int off[16];
    for (int k = 0; k < 16; ++k) off[k] = kCircle[k][1] * w + kCircle[k][0];
    for (int y = border; y < h - border; ++y) {
        const uint8_t* row = &img.px[static_cast<size_t>(y) * w];
        for (int x = border; x < w - border; ++x) {
            const uint8_t* p = row + x;
            int c = *p, hi = c + t, lo = c - t;
            // quick rejection: of pixels 0, 4, 8, 12 at least 3 must pass for a 9-arc... use 2 of 1/9 and 5/13
            int a0 = p[off[0]], a8 = p[off[8]];
            if ((a0 <= hi && a0 >= lo) && (a8 <= hi && a8 >= lo)) continue;
            int a4 = p[off[4]], a12 = p[off[12]];
            int nb = (a0 > hi) + (a4 > hi) + (a8 > hi) + (a12 > hi), nd = (a0 < lo) + (a4 < lo) + (a8 < lo) + (a12 < lo);
            if (nb < 2 && nd < 2) continue;
            int v[16];
            for (int k = 0; k < 16; ++k) v[k] = p[off[k]];
            bool corner = false;
            for (int sign = 0; sign < 2 && !corner; ++sign) {
                int run = 0;
                for (int k = 0; k < 16 + 9; ++k) {
                    int q = v[k & 15];
                    bool pass = sign == 0 ? q > hi : q < lo;
                    run = pass ? run + 1 : 0;
                    if (run >= 9) {
                        corner = true;
                        break;
                    }
                }
            }
            if (!corner) continue;
            int s = 0;
            for (int k = 0; k < 16; ++k) {
                int d = std::abs(v[k] - c) - t;
                if (d > 0) s += d;
            }
            score[static_cast<size_t>(y) * w + x] = s;
        }
    }
    std::vector<Corner> out;
    for (int y = border; y < h - border; ++y)
        for (int x = border; x < w - border; ++x) {
            int s = score[static_cast<size_t>(y) * w + x];
            if (!s) continue;
            bool max = true;
            for (int dy = -1; dy <= 1 && max; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    int o = score[static_cast<size_t>(y + dy) * w + x + dx];
                    if (o > s || (o == s && (dy < 0 || (dy == 0 && dx < 0)))) {
                        max = false;
                        break;
                    }
                }
            if (max) out.push_back({static_cast<float>(x), static_cast<float>(y), s});
        }
    return out;
}

std::vector<Corner> select_spread(std::vector<Corner> corners, const std::vector<Vec2>& existing, int w, int h, double min_dist, size_t max_new) {
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) { return a.score > b.score; });
    // occupancy grid of cell size min_dist: a cell can hold one point
    int cs = std::max(1, static_cast<int>(min_dist));
    int gw = w / cs + 1, gh = h / cs + 1;
    std::vector<uint8_t> occ(static_cast<size_t>(gw) * gh, 0);
    auto mark = [&](double x, double y) {
        int gx = static_cast<int>(x) / cs, gy = static_cast<int>(y) / cs;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int X = gx + dx, Y = gy + dy;
                if (X >= 0 && Y >= 0 && X < gw && Y < gh) occ[static_cast<size_t>(Y) * gw + X] = 1;
            }
    };
    for (const Vec2& e : existing) {
        int gx = static_cast<int>(e[0]) / cs, gy = static_cast<int>(e[1]) / cs;
        if (gx >= 0 && gy >= 0 && gx < gw && gy < gh) occ[static_cast<size_t>(gy) * gw + gx] = 1;
    }
    std::vector<Corner> out;
    for (const Corner& c : corners) {
        if (out.size() >= max_new) break;
        int gx = static_cast<int>(c.x) / cs, gy = static_cast<int>(c.y) / cs;
        if (occ[static_cast<size_t>(gy) * gw + gx]) continue;
        // also keep away from existing points in neighboring cells
        bool near = false;
        for (const Vec2& e : existing)
            if (std::fabs(e[0] - c.x) < min_dist && std::fabs(e[1] - c.y) < min_dist) {
                near = true;
                break;
            }
        if (near) continue;
        out.push_back(c);
        mark(c.x, c.y);
    }
    return out;
}

void klt_track(const std::vector<ImageF>& A, const std::vector<ImageF>& B, const std::vector<Vec2>& pts, std::vector<Vec2>& guess, std::vector<uint8_t>& ok,
               const KltParams& p) {
    int L = std::min<int>({p.levels, static_cast<int>(A.size()), static_cast<int>(B.size())});
    int hw = p.half_window, n = 2 * hw + 1;
    ok.assign(pts.size(), 0);
    std::vector<float> T(static_cast<size_t>(n) * n), Gx(T.size()), Gy(T.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        // displacement estimate carried down the pyramid
        double gx = (guess[i][0] - pts[i][0]) / (1 << (L - 1)), gy = (guess[i][1] - pts[i][1]) / (1 << (L - 1));
        bool good = true;
        for (int l = L - 1; l >= 0; --l) {
            const ImageF& a = A[l];
            const ImageF& b = B[l];
            double sc = 1.0 / (1 << l);
            double px = pts[i][0] * sc, py = pts[i][1] * sc;
            if (px - hw - 1 < 0 || py - hw - 1 < 0 || px + hw + 1 >= a.w - 1 || py + hw + 1 >= a.h - 1) {
                good = l > 0;  // too close to the border at a coarse level: skip the level
                if (l > 0) {
                    gx *= 2;
                    gy *= 2;
                    continue;
                }
                break;
            }
            // template and its gradient
            double sxx = 0, sxy = 0, syy = 0;
            for (int dy = -hw, k = 0; dy <= hw; ++dy)
                for (int dx = -hw; dx <= hw; ++dx, ++k) {
                    float x = static_cast<float>(px + dx), y = static_cast<float>(py + dy);
                    T[k] = a.sample(x, y);
                    Gx[k] = 0.5f * (a.sample(x + 1, y) - a.sample(x - 1, y));
                    Gy[k] = 0.5f * (a.sample(x, y + 1) - a.sample(x, y - 1));
                    sxx += Gx[k] * Gx[k];
                    sxy += Gx[k] * Gy[k];
                    syy += Gy[k] * Gy[k];
                }
            double det = sxx * syy - sxy * sxy;
            if (det < 1e-6 * n * n) {
                good = false;
                break;
            }
            for (int it = 0; it < p.iterations; ++it) {
                double qx = px + gx, qy = py + gy;
                if (qx - hw < 0 || qy - hw < 0 || qx + hw >= b.w - 1 || qy + hw >= b.h - 1) {
                    good = false;
                    break;
                }
                double bx = 0, by = 0;
                for (int dy = -hw, k = 0; dy <= hw; ++dy)
                    for (int dx = -hw; dx <= hw; ++dx, ++k) {
                        double e = b.sample(static_cast<float>(qx + dx), static_cast<float>(qy + dy)) - T[k];
                        bx += e * Gx[k];
                        by += e * Gy[k];
                    }
                double ux = -(syy * bx - sxy * by) / det, uy = -(-sxy * bx + sxx * by) / det;
                gx += ux;
                gy += uy;
                if (ux * ux + uy * uy < p.epsilon * p.epsilon) break;
            }
            if (!good) break;
            if (l > 0) {
                gx *= 2;
                gy *= 2;
            }
        }
        guess[i] = Vec2{pts[i][0] + gx, pts[i][1] + gy};
        const ImageF& b0 = B[0];
        ok[i] = good && guess[i][0] >= 1 && guess[i][1] >= 1 && guess[i][0] < b0.w - 2 && guess[i][1] < b0.h - 2;
    }
}

// ---- ORB --------------------------------------------------------------------

namespace {
struct Pattern {
    int8_t p[256][4];
    Pattern() {
        // a fixed random sampling pattern inside a 31x31 patch (Gaussian, sigma 31/5)
        std::mt19937 rng(12345);
        std::normal_distribution<double> N(0, 31.0 / 5);
        for (int i = 0; i < 256; ++i)
            for (int k = 0; k < 4; ++k) {
                double v;
                do {
                    v = std::round(N(rng));
                } while (std::fabs(v) > 13);
                p[i][k] = static_cast<int8_t>(v);
            }
    }
};
const Pattern& pattern() {
    static Pattern p;
    return p;
}
}  // namespace

std::vector<OrbFeature> orb_describe(const Image& img, const std::vector<Corner>& corners) {
    std::vector<OrbFeature> out;
    // smoothed image for the binary tests
    Image sm = img;
    for (int y = 2; y < img.h - 2; ++y)
        for (int x = 2; x < img.w - 2; ++x) {
            int s = 0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) s += img.at(x + dx, y + dy);
            sm.at(x, y) = static_cast<uint8_t>(s / 25);
        }
    const Pattern& pat = pattern();
    for (const Corner& c : corners) {
        int x = static_cast<int>(c.x), y = static_cast<int>(c.y);
        if (x < 20 || y < 20 || x >= img.w - 20 || y >= img.h - 20) continue;
        // orientation by the intensity centroid in a radius-15 disc
        double m01 = 0, m10 = 0;
        for (int dy = -15; dy <= 15; ++dy)
            for (int dx = -15; dx <= 15; ++dx)
                if (dx * dx + dy * dy <= 225) {
                    double v = img.at(x + dx, y + dy);
                    m10 += dx * v;
                    m01 += dy * v;
                }
        double ang = std::atan2(m01, m10), ca = std::cos(ang), sa = std::sin(ang);
        OrbFeature f{c.x, c.y, static_cast<float>(ang), {0, 0, 0, 0}};
        for (int i = 0; i < 256; ++i) {
            auto rot = [&](int a, int b, int& rx, int& ry) {
                rx = static_cast<int>(std::lround(ca * a - sa * b));
                ry = static_cast<int>(std::lround(sa * a + ca * b));
            };
            int ax, ay, bx, by;
            rot(pat.p[i][0], pat.p[i][1], ax, ay);
            rot(pat.p[i][2], pat.p[i][3], bx, by);
            if (sm.at(x + ax, y + ay) < sm.at(x + bx, y + by)) f.d[i >> 6] |= uint64_t(1) << (i & 63);
        }
        out.push_back(f);
    }
    return out;
}

}  // namespace rover
