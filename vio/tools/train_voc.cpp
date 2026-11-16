// rover-voc: train the visual vocabulary for loop closure.
//
//   rover-voc OUT.voc SEQUENCE_DIR [SEQUENCE_DIR ...]
//
// ORB features of every 10th left image of the given sequences, clustered
// into a 10-way tree of depth 4; idf weights from the same images.
#include <cmath>
#include <cstdio>
#include <set>

#include "rover/euroc.hpp"
#include "rover/image.hpp"
#include "rover/loop.hpp"

using namespace rover;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: rover-voc OUT.voc SEQUENCE_DIR...\n");
        return 2;
    }
    std::vector<Desc> all;
    std::vector<std::vector<Desc>> per_image;
    for (int a = 2; a < argc; ++a) {
        EurocSequence s = EurocSequence::load(argv[a]);
        for (size_t k = 0; k < s.frames.size(); k += 10) {
            auto f = loop_features(read_png(s.frames[k].left));
            std::vector<Desc> d;
            for (auto& o : f) d.push_back(o.d);
            all.insert(all.end(), d.begin(), d.end());
            per_image.push_back(d);
        }
        std::fprintf(stderr, "%s: %zu descriptors so far\n", argv[a], all.size());
    }
    Vocabulary voc;
    voc.train(all, 10, 4);
    std::vector<int> df(voc.words(), 0);
    for (auto& img : per_image) {
        std::set<int> seen;
        for (auto& d : img) seen.insert(voc.word(d));
        for (int w : seen) df[w]++;
    }
    std::vector<double> idf(voc.words());
    for (int w = 0; w < voc.words(); ++w) idf[w] = std::log(static_cast<double>(per_image.size()) / std::max(1, df[w]));
    voc.set_idf(idf);
    if (!voc.save(argv[1])) {
        std::fprintf(stderr, "cannot write %s\n", argv[1]);
        return 1;
    }
    std::printf("{\"images\": %zu, \"descriptors\": %zu, \"words\": %d}\n", per_image.size(), all.size(), voc.words());
    return 0;
}
