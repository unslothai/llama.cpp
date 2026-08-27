#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

// GLM-5-Next / GLM-5.3-Flash, hparams as loaded by clip.cpp for PROJECTOR_TYPE_GLM5NEXT
static clip_hparams glm5next_hparams() {
    clip_hparams hparams;
    hparams.patch_size = 14;
    hparams.n_merge    = 2;
    hparams.set_limit_image_tokens(16, 8000);
    return hparams;
}

// set_limit_image_tokens takes token counts; for a still image the reference's temporal factors cancel,
// leaving factor**2 == 28*28
MAKE_TEST(test_image_preprocessor_glm5next_budget) {
    const clip_hparams hparams = glm5next_hparams();

    t.assert_equal("image_min_pixels", 16   * 28 * 28, hparams.image_min_pixels);
    t.assert_equal("image_max_pixels", 8000 * 28 * 28, hparams.image_max_pixels);
}

MAKE_TEST(test_image_preprocessor_glm5next_resize) {
    const clip_hparams hparams = glm5next_hparams();

    struct test_case {
        clip_image_size input;
        clip_image_size canvas;  // padded output
        clip_image_size content; // resized image inside the canvas
        int n_tokens;
    };

    // expected values come from Glm5NextImageProcessor.resize in
    // adapt_zips/extract_0826/image_processing_glm5_next.py
    const std::vector<test_case> cases = {
        // inside the budget
        { {   224,   224 }, {   224,   224 }, {   224,   224 },   64 },
        { {   448,   448 }, {   448,   448 }, {   448,   448 },  256 },
        { {  1024,  1024 }, {  1036,  1036 }, {  1024,  1024 }, 1369 },
        // the 16-token floor: 112x112 is exactly 16 tokens
        { {   112,   112 }, {   112,   112 }, {   112,   112 },   16 },
        { {   111,   111 }, {   112,   112 }, {   111,   111 },   16 },
        { {   113,   113 }, {   140,   140 }, {   113,   113 },   25 },
        { {    56,    56 }, {   112,   112 }, {   112,   112 },   16 },
        { {    50,    50 }, {   112,   112 }, {   112,   112 },   16 },
        { {    28,    28 }, {   112,   112 }, {   112,   112 },   16 },
        { {     1,     1 }, {   112,   112 }, {   112,   112 },   16 },
        // the 8000-token ceiling: 2492x2492 still fits, 2520x2520 does not
        { {  2492,  2492 }, {  2492,  2492 }, {  2492,  2492 }, 7921 },
        { {  2493,  2493 }, {  2492,  2492 }, {  2492,  2492 }, 7921 },
        { {  2504,  2504 }, {  2492,  2492 }, {  2492,  2492 }, 7921 },
        { {  2520,  2520 }, {  2492,  2492 }, {  2492,  2492 }, 7921 },
        { {  4000,  4000 }, {  2492,  2492 }, {  2492,  2492 }, 7921 },
        // wide and tall
        { {   800,   600 }, {   812,   616 }, {   800,   600 },  638 },
        { {   600,   800 }, {   616,   812 }, {   600,   800 },  638 },
        { {  1920,   480 }, {  1932,   504 }, {  1920,   480 }, 1242 },
        { {   480,  1920 }, {   504,  1932 }, {   480,  1920 }, 1242 },
        { {  1920,  1080 }, {  1932,  1092 }, {  1920,  1080 }, 2691 },
        { {  1080,  1920 }, {  1092,  1932 }, {  1080,  1920 }, 2691 },
        // extreme aspect ratios, single-pixel edges survive the resize
        { {  4000,    16 }, {  4004,    28 }, {  4000,    16 },  143 },
        { {    16,  4000 }, {    28,  4004 }, {    16,  4000 },  143 },
        { {  5000,     1 }, {  5012,    28 }, {  5012,     1 },  179 },
        { {     1,  5000 }, {    28,  5012 }, {     1,  5012 },  179 },
        { { 12000,   100 }, { 12012,   112 }, { 12000,   100 }, 1716 },
        { {   100, 12000 }, {   112, 12012 }, {   100, 12000 }, 1716 },
        // over budget, neither edge a multiple of 28
        { {  4007,  3001 }, {  2884,  2156 }, {  2878,  2156 }, 7931 },
        { {  3001,  4007 }, {  2156,  2884 }, {  2156,  2878 }, 7931 },
        { {  3333,  5000 }, {  2044,  3052 }, {  2034,  3052 }, 7957 },
        { {  2729,  2731 }, {  2492,  2492 }, {  2490,  2492 }, 7921 },
        // the 0826 binary search and a Qwen-style smart_resize disagree on all of these, so a regression
        // to the old dynamic-size preprocessor cannot pass
        { {  4618,  2282 }, {  3528,  1764 }, {  3528,  1743 }, 7938 }, // smart_resize: 3556x1736
        { {  2794,  6096 }, {  1708,  3668 }, {  1681,  3668 }, 7991 }, // smart_resize: 1680x3696
        { {  8858,  1315 }, {  6412,   952 }, {  6412,   951 }, 7786 }, // smart_resize: 6496x952
        { {  1350,  5856 }, {  1204,  5208 }, {  1200,  5208 }, 7998 }, // smart_resize: 1176x5208
        { {  2134,  1472 }, {  2156,  1484 }, {  2134,  1472 }, 4081 }, // smart_resize: 2128x1484
        { {  1021,  4268 }, {  1036,  4284 }, {  1021,  4268 }, 5661 }, // smart_resize: 1008x4256
    };

    auto fmt = [](const clip_image_size & s) {
        return std::to_string(s.width) + "x" + std::to_string(s.height);
    };

    for (const auto & tc : cases) {
        const auto geo  = mtmd_image_preprocessor_glm5next::get_geometry(hparams, tc.input);
        const auto name = " for " + fmt(tc.input);

        t.assert_equal("canvas"  + name, fmt(tc.canvas),  fmt(geo.canvas));
        t.assert_equal("content" + name, fmt(tc.content), fmt(geo.content));

        // mirrors clip_n_output_tokens for PROJECTOR_TYPE_GLM5NEXT
        const int n_tokens = (geo.canvas.width  / (hparams.patch_size * hparams.n_merge))
                           * (geo.canvas.height / (hparams.patch_size * hparams.n_merge));
        t.assert_equal("n_tokens" + name, tc.n_tokens, n_tokens);

        t.assert_true("content fits the canvas" + name,
            geo.content.width <= geo.canvas.width && geo.content.height <= geo.canvas.height);
        t.assert_true("canvas is within the token budget" + name,
            geo.canvas.width * geo.canvas.height <= hparams.image_max_pixels);
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_input_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
