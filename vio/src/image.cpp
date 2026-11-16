#include "rover/image.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace rover {

namespace {

struct BitReader {
    const uint8_t* p;
    size_t n, pos = 0;
    uint32_t bitbuf = 0;
    int bitcnt = 0;
    int bit() {
        if (bitcnt == 0) {
            if (pos >= n) throw std::runtime_error("inflate: unexpected end of data");
            bitbuf = p[pos++];
            bitcnt = 8;
        }
        int b = bitbuf & 1;
        bitbuf >>= 1;
        --bitcnt;
        return b;
    }
    uint32_t bits(int k) {
        uint32_t v = 0;
        for (int i = 0; i < k; ++i) v |= static_cast<uint32_t>(bit()) << i;
        return v;
    }
    void align() { bitcnt = 0; }
};

/// canonical Huffman decoding table: counts per length and symbols in order
struct Huffman {
    uint16_t count[16] = {0};
    std::vector<uint16_t> symbol;
    void build(const uint8_t* lengths, int n) {
        std::memset(count, 0, sizeof count);
        for (int i = 0; i < n; ++i) count[lengths[i]]++;
        count[0] = 0;
        uint16_t offs[16];
        offs[1] = 0;
        for (int l = 1; l < 15; ++l) offs[l + 1] = offs[l] + count[l];
        symbol.assign(n, 0);
        for (int i = 0; i < n; ++i)
            if (lengths[i]) symbol[offs[lengths[i]]++] = static_cast<uint16_t>(i);
    }
    int decode(BitReader& br) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; ++len) {
            code |= br.bit();
            int c = count[len];
            if (code - c < first) return symbol[index + (code - first)];
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
        }
        throw std::runtime_error("inflate: bad Huffman code");
    }
};

const uint16_t kLenBase[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const uint16_t kLenExtra[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const uint16_t kDistBase[] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
const uint16_t kDistExtra[] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

void inflate_block(BitReader& br, std::vector<uint8_t>& out, const Huffman& lit, const Huffman& dist) {
    while (true) {
        int sym = lit.decode(br);
        if (sym < 256) {
            out.push_back(static_cast<uint8_t>(sym));
        } else if (sym == 256) {
            return;
        } else {
            sym -= 257;
            if (sym >= 29) throw std::runtime_error("inflate: bad length symbol");
            int len = kLenBase[sym] + static_cast<int>(br.bits(kLenExtra[sym]));
            int ds = dist.decode(br);
            if (ds >= 30) throw std::runtime_error("inflate: bad distance symbol");
            size_t d = kDistBase[ds] + br.bits(kDistExtra[ds]);
            if (d > out.size()) throw std::runtime_error("inflate: distance too far back");
            size_t start = out.size() - d;
            for (int i = 0; i < len; ++i) out.push_back(out[start + i]);
        }
    }
}

}  // namespace

std::vector<uint8_t> zlib_inflate(const uint8_t* data, size_t n) {
    if (n < 2 || (data[0] & 0x0F) != 8 || ((data[0] << 8) | data[1]) % 31 != 0) throw std::runtime_error("zlib: bad header");
    BitReader br{data + 2, n - 2};
    std::vector<uint8_t> out;
    out.reserve(n * 4);
    Huffman fixed_lit, fixed_dist;
    {
        uint8_t l[288];
        for (int i = 0; i < 144; ++i) l[i] = 8;
        for (int i = 144; i < 256; ++i) l[i] = 9;
        for (int i = 256; i < 280; ++i) l[i] = 7;
        for (int i = 280; i < 288; ++i) l[i] = 8;
        fixed_lit.build(l, 288);
        uint8_t d[30];
        for (int i = 0; i < 30; ++i) d[i] = 5;
        fixed_dist.build(d, 30);
    }
    bool last = false;
    while (!last) {
        last = br.bit();
        int type = static_cast<int>(br.bits(2));
        if (type == 0) {
            br.align();
            if (br.pos + 4 > br.n) throw std::runtime_error("inflate: truncated stored block");
            uint16_t len = br.p[br.pos] | (br.p[br.pos + 1] << 8);
            br.pos += 4;
            if (br.pos + len > br.n) throw std::runtime_error("inflate: truncated stored block");
            out.insert(out.end(), br.p + br.pos, br.p + br.pos + len);
            br.pos += len;
        } else if (type == 1) {
            inflate_block(br, out, fixed_lit, fixed_dist);
        } else if (type == 2) {
            int hlit = static_cast<int>(br.bits(5)) + 257, hdist = static_cast<int>(br.bits(5)) + 1, hclen = static_cast<int>(br.bits(4)) + 4;
            static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            uint8_t cl[19] = {0};
            for (int i = 0; i < hclen; ++i) cl[order[i]] = static_cast<uint8_t>(br.bits(3));
            Huffman clh;
            clh.build(cl, 19);
            uint8_t lens[320] = {0};
            int k = 0;
            while (k < hlit + hdist) {
                int sym = clh.decode(br);
                if (sym < 16) {
                    lens[k++] = static_cast<uint8_t>(sym);
                } else {
                    int rep = 0;
                    uint8_t v = 0;
                    if (sym == 16) {
                        if (k == 0) throw std::runtime_error("inflate: repeat with no previous length");
                        v = lens[k - 1];
                        rep = 3 + static_cast<int>(br.bits(2));
                    } else if (sym == 17) {
                        rep = 3 + static_cast<int>(br.bits(3));
                    } else {
                        rep = 11 + static_cast<int>(br.bits(7));
                    }
                    if (k + rep > hlit + hdist) throw std::runtime_error("inflate: too many lengths");
                    while (rep--) lens[k++] = v;
                }
            }
            Huffman lit, dist;
            lit.build(lens, hlit);
            dist.build(lens + hlit, hdist);
            inflate_block(br, out, lit, dist);
        } else {
            throw std::runtime_error("inflate: bad block type");
        }
    }
    return out;
}

static uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }

Image decode_png(const std::vector<uint8_t>& f) {
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (f.size() < 8 || std::memcmp(f.data(), sig, 8) != 0) throw std::runtime_error("png: bad signature");
    size_t pos = 8;
    int w = 0, h = 0, depth = 0, ctype = 0, interlace = 0;
    std::vector<uint8_t> idat;
    while (pos + 8 <= f.size()) {
        uint32_t len = be32(&f[pos]);
        const char* type = reinterpret_cast<const char*>(&f[pos + 4]);
        if (pos + 12 + len > f.size()) throw std::runtime_error("png: truncated chunk");
        const uint8_t* d = &f[pos + 8];
        if (std::memcmp(type, "IHDR", 4) == 0) {
            w = static_cast<int>(be32(d));
            h = static_cast<int>(be32(d + 4));
            depth = d[8];
            ctype = d[9];
            interlace = d[12];
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), d, d + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + len;
    }
    if (depth != 8 || interlace != 0) throw std::runtime_error("png: only 8-bit, non-interlaced images");
    int channels = ctype == 0 ? 1 : ctype == 2 ? 3 : ctype == 4 ? 2 : ctype == 6 ? 4 : 0;
    if (!channels) throw std::runtime_error("png: unsupported color type");
    std::vector<uint8_t> raw = zlib_inflate(idat.data(), idat.size());
    size_t stride = static_cast<size_t>(w) * channels;
    if (raw.size() < (stride + 1) * h) throw std::runtime_error("png: not enough image data");
    std::vector<uint8_t> cur(stride), prev(stride, 0);
    Image img;
    img.w = w;
    img.h = h;
    img.px.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* line = &raw[y * (stride + 1)];
        int filter = line[0];
        const uint8_t* s = line + 1;
        for (size_t i = 0; i < stride; ++i) {
            int a = i >= static_cast<size_t>(channels) ? cur[i - channels] : 0;
            int b = prev[i];
            int c = i >= static_cast<size_t>(channels) ? prev[i - channels] : 0;
            int v = s[i];
            switch (filter) {
                case 0: break;
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (a + b) >> 1; break;
                case 4: {
                    int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                    v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                    break;
                }
                default: throw std::runtime_error("png: bad filter");
            }
            cur[i] = static_cast<uint8_t>(v);
        }
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = &cur[static_cast<size_t>(x) * channels];
            img.px[static_cast<size_t>(y) * w + x] = channels <= 2 ? p[0] : static_cast<uint8_t>((299 * p[0] + 587 * p[1] + 114 * p[2]) / 1000);
        }
        std::swap(cur, prev);
    }
    return img;
}

Image read_png(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> f((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return decode_png(f);
}

void write_pgm(const std::string& path, const Image& img) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P5\n%d %d\n255\n", img.w, img.h);
    std::fwrite(img.px.data(), 1, img.px.size(), f);
    std::fclose(f);
}

ImageF to_float(const Image& img) {
    ImageF f;
    f.w = img.w;
    f.h = img.h;
    f.px.resize(img.px.size());
    for (size_t i = 0; i < img.px.size(); ++i) f.px[i] = img.px[i];
    return f;
}

ImageF downsample(const ImageF& img) {
    // separable [1 4 6 4 1]/16 blur, then every second pixel
    int w = img.w, h = img.h;
    ImageF tmp;
    tmp.w = w;
    tmp.h = h;
    tmp.px.resize(img.px.size());
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float s = img.at(clampi(x - 2, 0, w - 1), y) + 4 * img.at(clampi(x - 1, 0, w - 1), y) + 6 * img.at(x, y) + 4 * img.at(clampi(x + 1, 0, w - 1), y) +
                      img.at(clampi(x + 2, 0, w - 1), y);
            tmp.at(x, y) = s / 16;
        }
    ImageF out;
    out.w = w / 2;
    out.h = h / 2;
    out.px.resize(static_cast<size_t>(out.w) * out.h);
    for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.w; ++x) {
            int sx = 2 * x, sy = 2 * y;
            float s = tmp.at(sx, clampi(sy - 2, 0, h - 1)) + 4 * tmp.at(sx, clampi(sy - 1, 0, h - 1)) + 6 * tmp.at(sx, sy) + 4 * tmp.at(sx, clampi(sy + 1, 0, h - 1)) +
                      tmp.at(sx, clampi(sy + 2, 0, h - 1));
            out.at(x, y) = s / 16;
        }
    return out;
}

std::vector<ImageF> pyramid(const Image& img, int levels) {
    std::vector<ImageF> p;
    p.push_back(to_float(img));
    for (int l = 1; l < levels; ++l) p.push_back(downsample(p.back()));
    return p;
}

}  // namespace rover
