// [TAG_PREEMPT] KV position and token count differ with M-RoPE media, so test the rewind conversion on a hand-built image chunk.

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

// the mtmd_input_chunk_save() wire format, written by hand so a chunk loads without a context.
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

static mtmd::input_chunk_ptr make_image_chunk(uint32_t nx, uint32_t ny) {
    chunk_writer w;

    w.put<uint64_t>(2);
    w.put<uint32_t>(MTMD_INPUT_CHUNK_TYPE_IMAGE);
    w.put<uint64_t>(0);
    w.put<uint8_t>(1);
    w.put<uint32_t>(nx);
    w.put<uint32_t>(ny);
    w.put<uint32_t>(1);
    w.put<uint32_t>(0);
    w.put<uint32_t>(1);
    w.put_str("test-image");
    w.put<uint8_t>(0);
    w.put<uint64_t>(1);
    w.put<uint8_t>(0);
    w.put<uint8_t>(0);
    w.put<int32_t>(0);
    w.put<int32_t>(1);
    w.put<int32_t>(1);
    w.put<uint8_t>(0);

    mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(w.buf.data(), w.buf.size()));

    assert(chunk && "the serialized image chunk was rejected");

    return chunk;
}

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

    {
        server_tokens prompt = make_prompt(40, chunk.get());

        assert(prompt.size()     == 326);
        assert(prompt.pos_next() == 86);

        const llama_pos pos_cached = 86;
        const size_t    n_cached   = prompt.size_up_to_pos(pos_cached);

        assert(n_cached == 326);
        assert(prompt.pos_next(n_cached) == pos_cached);

        bool threw = false;

        try {
            prompt.keep_first((size_t) pos_cached);
        } catch (const std::exception &) {
            threw = true;
        }

        assert(threw && "a position used as a token count cuts the image in half");
    }

    {
        server_tokens prompt = make_prompt(300, chunk.get());

        assert(prompt.size()     == 586);
        assert(prompt.pos_next() == 346);

        const llama_pos pos_cached = 106;
        const size_t    n_cached   = prompt.size_up_to_pos(pos_cached);

        assert(n_cached == 346);
        assert(prompt.pos_next(n_cached) == pos_cached);

        prompt.keep_first(n_cached);

        assert(prompt.size()     == 346);
        assert(prompt.pos_next() == pos_cached);
    }

    {
        server_tokens prompt = make_prompt(0, chunk.get());

        assert(prompt.size_up_to_pos(10) == 10);
        assert(prompt.pos_next(10) == 10);

        assert(prompt.size_up_to_pos(26) == 266);
        assert(prompt.pos_next(266) == 26);
    }

    {
        server_tokens prompt = make_prompt(0, chunk.get());

        const llama_pos pos_cached = 20;

        size_t n_cached = prompt.size_up_to_pos(pos_cached);

        assert(n_cached == 266);

        while (n_cached > 0 && prompt.pos_next(n_cached) > pos_cached) {
            n_cached--;
        }

        assert(n_cached == 10);
        assert(prompt.pos_next(n_cached) == 10);

        prompt.keep_first(n_cached);
        assert(prompt.size() == 10);
    }

    {
        server_tokens prompt = make_prompt(4, chunk.get());

        const llama_pos pos_cached = 0;

        assert(prompt.size_up_to_pos(pos_cached) == 1);
        assert((pos_cached > 0 ? prompt.size_up_to_pos(pos_cached) : 0) == 0);
    }

    printf("%s: all tests passed\n", __func__);

    return 0;
}
