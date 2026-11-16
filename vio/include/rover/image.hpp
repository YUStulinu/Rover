// Images: 8-bit grayscale and float buffers, a PNG decoder (DEFLATE inflate
// and the five PNG filters, written for EuRoC's 8-bit grayscale images but
// handling gray, RGB and alpha at 8 bits), and a PGM writer for debugging.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rover {

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
    uint8_t at(int x, int y) const { return px[static_cast<size_t>(y) * w + x]; }
    uint8_t& at(int x, int y) { return px[static_cast<size_t>(y) * w + x]; }
};

struct ImageF {
    int w = 0, h = 0;
    std::vector<float> px;
    float at(int x, int y) const { return px[static_cast<size_t>(y) * w + x]; }
    float& at(int x, int y) { return px[static_cast<size_t>(y) * w + x]; }
    /// bilinear interpolation (x, y inside [0, w-1] x [0, h-1])
    float sample(float x, float y) const {
        int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
        if (x0 >= w - 1) x0 = w - 2;
        if (y0 >= h - 1) y0 = h - 2;
        float ax = x - x0, ay = y - y0;
        const float* p = &px[static_cast<size_t>(y0) * w + x0];
        return (1 - ay) * ((1 - ax) * p[0] + ax * p[1]) + ay * ((1 - ax) * p[w] + ax * p[w + 1]);
    }
};

/// Inflate a zlib stream (RFC 1950/1951). Throws std::runtime_error on bad data.
std::vector<uint8_t> zlib_inflate(const uint8_t* data, size_t n);

/// Decode a PNG file into 8-bit gray (color images are converted). Throws on error.
Image read_png(const std::string& path);
Image decode_png(const std::vector<uint8_t>& file);

void write_pgm(const std::string& path, const Image& img);

ImageF to_float(const Image& img);
/// half resolution with a 2x2 box filter after a 5-tap Gaussian
ImageF downsample(const ImageF& img);
std::vector<ImageF> pyramid(const Image& img, int levels);

}  // namespace rover
