// The PNG decoder (own inflate and filters) against a file compressed by
// zlib at level 9 through PIL: every pixel must match.
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "rover/image.hpp"
#include "png_fixture.inc"

int main() {
    int fails = 0;
    rover::Image img = rover::decode_png(std::vector<uint8_t>(kPng8, kPng8 + sizeof kPng8));
    if (img.w != kW || img.h != kH) {
        std::printf("FAIL size %dx%d\n", img.w, img.h);
        return 1;
    }
    for (int i = 0; i < kW * kH; ++i) fails += img.px[i] != kPixels[i];
    std::printf("8-bit gray: %d of %d pixels differ\n", fails, kW * kH);
    bool threw = false;
    try {
        rover::decode_png(std::vector<uint8_t>(kPng16, kPng16 + sizeof kPng16));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::printf("16-bit rejected: %s\n", threw ? "yes" : "no");
    std::vector<uint8_t> bad(kPng8, kPng8 + sizeof kPng8);
    bad[60] ^= 0x5a;  // corrupt the compressed stream
    bool threw2 = false;
    try {
        auto b = rover::decode_png(bad);
        threw2 = b.w != kW;  // a corrupt stream must throw (or at least not pass silently as the same image)
        int diff = 0;
        for (int i = 0; i < kW * kH; ++i) diff += b.px[i] != kPixels[i];
        threw2 = threw2 || diff > 0;
    } catch (const std::exception&) {
        threw2 = true;
    }
    std::printf("corruption detected: %s\n", threw2 ? "yes" : "no");
    return (fails == 0 && threw && threw2) ? 0 : 1;
}
