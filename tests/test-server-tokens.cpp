// [TAG_PREEMPT] the server converts between a KV position and a token count when it rewinds a slot to
// what the cache holds. With M-RoPE media the two differ, so the conversion is exercised here on a
// hand-built image chunk, without a model.

#include "server-common.h"

#include "mtmd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

// the wire format of mtmd_input_chunk_save(), which needs a context to produce a chunk; written here so
// that a chunk of a known shape can be loaded without one
struct chunk_writer {
    std::vector<char> buf;

    template <typename T> void put(T v) {
        const char * p = reinterpret_cast<const char *>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    void put_str(const std::string & s) {
        put<uint64_t>(s.size());
        buf.insert(buf.end(), s.begin(), s.end());
    }
};

// nx*ny tokens of one image, max(nx, ny) positions under M-RoPE
static mtmd::input_chunk_ptr make_image_chunk(uint32_t nx, uint32_t ny) {
    chunk_writer w;

    w.put<uint64_t>(1);                          // MTMD_SERIALIZATION_VERSION
    w.put<uint32_t>(MTMD_INPUT_CHUNK_TYPE_IMAGE);
    w.put<uint64_t>(0);                          // tokens_text
    w.put<uint8_t>(1);                           // tokens_image follows
    w.put<uint32_t>(nx);
    w.put<uint32_t>(ny);
    w.put<uint32_t>(1);                          // MTMD_POS_TYPE_MROPE
    w.put<uint32_t>(0);                          // image_idx
    w.put<uint32_t>(1);                          // n_temporal_merge
    w.put_str("test-image");                     // id
    w.put<uint8_t>(0);                           // batch_f32.is_audio
    w.put<uint64_t>(1);                          // one entry
    w.put<uint8_t>(0);                           // entry.add_viewsep
    w.put<uint8_t>(0);                           // entry.add_newline
    w.put<int32_t>(1);                           // entry.nx
    w.put<int32_t>(1);                           // entry.ny
    w.put<uint8_t>(0);                           // no tokens_audio

    mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(w.buf.data(), w.buf.size()));

    assert(chunk && "the serialized image chunk was rejected");

    return chunk;
}

// 10 text tokens, an image of 256 tokens and 16 positions, 20 text tokens, then n_gen generated tokens
static server_tokens make_prompt(size_t n_gen, const mtmd_input_chunk * chunk) {
    server_tokens res;

    res.has_mtmd = true;

    for (size_t i = 0; i < 10; ++i) {
        res.push_back((llama_token) (100 + i));
    }

    res.push_back(chunk);

    for (size_t i = 0; i < 20; ++i) {
        res.push_back((llama_token) (200 + i));
    }

    for (size_t i = 0; i < n_gen; ++i) {
        res.push_back((llama_token) (300 + i));
    }

    return res;
}

int main() {
    const mtmd::input_chunk_ptr chunk = make_image_chunk(16, 16);

    assert(mtmd_input_chunk_get_n_tokens(chunk.get()) == 256);
    assert(mtmd_input_chunk_get_n_pos   (chunk.get()) == 16);

    // a cut in the generated tail: the cache reports 86 positions, which is 326 tokens
    {
        server_tokens prompt = make_prompt(40, chunk.get());

        assert(prompt.size()     == 326);
        assert(prompt.pos_next() == 86);

        const llama_pos pos_cached = 86;
        const size_t    n_cached   = prompt.size_up_to_pos(pos_cached);

        assert(n_cached == 326);
        assert(prompt.pos_next(n_cached) == pos_cached);

        // the same number taken for a token count falls inside the image
        bool threw = false;

        try {
            prompt.keep_first((size_t) pos_cached);
        } catch (const std::exception &) {
            threw = true;
        }

        assert(threw && "a position used as a token count cuts the image in half");
    }

    // the same prompt with a longer tail, cut inside the generated tokens
    {
        server_tokens prompt = make_prompt(300, chunk.get());

        assert(prompt.size()     == 586);
        assert(prompt.pos_next() == 346);

        const llama_pos pos_cached = 106; // 10 text + 16 image + 20 text + 60 generated
        const size_t    n_cached   = prompt.size_up_to_pos(pos_cached);

        assert(n_cached == 346);
        assert(prompt.pos_next(n_cached) == pos_cached);

        prompt.keep_first(n_cached);

        assert(prompt.size()     == 346);
        assert(prompt.pos_next() == pos_cached);
    }

    // a cut before the image, and one at its first token: both are token boundaries
    {
        server_tokens prompt = make_prompt(0, chunk.get());

        assert(prompt.size_up_to_pos(10) == 10);
        assert(prompt.pos_next(10) == 10);

        // the image ends at position 26 and token 266
        assert(prompt.size_up_to_pos(26) == 266);
        assert(prompt.pos_next(266) == 26);
    }

    // a cut inside the image: the conversion cannot land there, and stepping back reaches the chunk's first token
    {
        server_tokens prompt = make_prompt(0, chunk.get());

        const llama_pos pos_cached = 20; // inside the image, which spans positions 10..25

        size_t n_cached = prompt.size_up_to_pos(pos_cached);

        assert(n_cached == 266); // rounded up to the whole chunk

        while (n_cached > 0 && prompt.pos_next(n_cached) > pos_cached) {
            n_cached--;
        }

        assert(n_cached == 10);
        assert(prompt.pos_next(n_cached) == 10);

        prompt.keep_first(n_cached); // would throw if it cut the image in half
        assert(prompt.size() == 10);
    }

    // an empty cache has to be handled by the caller: the walk always consumes its first token
    {
        server_tokens prompt = make_prompt(4, chunk.get());

        const llama_pos pos_cached = 0;

        assert(prompt.size_up_to_pos(pos_cached) == 1);
        assert((pos_cached > 0 ? prompt.size_up_to_pos(pos_cached) : 0) == 0);
    }

    printf("%s: all tests passed\n", __func__);

    return 0;
}
