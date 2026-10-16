// llama_moe_cache_materialize_mapped: tensors in memory-mapped file buffers (CPU_Mapped) are copied into one anonymous host buffer
// and repointed, the other tensors are left untouched (CPU only)

#include "../src/llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#    define TEST_HAVE_MMAP 1
#endif

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

int main() {
    ggml_init_params params = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));

    // two "expert" tensors of a quantized type and an f32 one, laid out like a model file region
    ggml_tensor * a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_0, 64, 3, 4);
    ggml_tensor * b = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32,  7, 5, 4);
    ggml_tensor * c = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 2); // in an anonymous buffer, must stay
    const size_t align = 64;
    const size_t off_a = 0;
    const size_t off_b = GGML_PAD(off_a + ggml_nbytes(a), align) + 4096; // a gap like the bytes of other tensors in the file
    const size_t size  = GGML_PAD(off_b + ggml_nbytes(b), align);

    std::vector<uint8_t> content(size);
    for (size_t i = 0; i < size; ++i) {
        content[i] = (uint8_t) (i*2654435761u >> 13);
    }

    void * region = nullptr;
#ifdef TEST_HAVE_MMAP
    char path[] = "/tmp/test-moe-cache-materialize-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(write(fd, content.data(), size) == (ssize_t) size);
    region = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0); // read-only file mapping, as llama_mmap does
    CHECK(region != MAP_FAILED);
    close(fd);
    unlink(path);
#else
    std::vector<uint8_t> storage(size + 4096);
    region = (void *) GGML_PAD((uintptr_t) storage.data(), 4096);
    memcpy(region, content.data(), size);
#endif

    ggml_backend_buffer_ptr mapped(ggml_backend_cpu_buffer_from_ptr(region, size));
    CHECK(strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(mapped.get())), "CPU_Mapped") == 0);
    ggml_backend_buffer_set_usage(mapped.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(ggml_backend_tensor_alloc(mapped.get(), a, (uint8_t *) region + off_a) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_tensor_alloc(mapped.get(), b, (uint8_t *) region + off_b) == GGML_STATUS_SUCCESS);

    ggml_backend_buffer_ptr anon(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(c)));
    CHECK(ggml_backend_tensor_alloc(anon.get(), c, ggml_backend_buffer_get_base(anon.get())) == GGML_STATUS_SUCCESS);
    void * c_data = c->data;

    // nothing to move
    {
        ggml_tensor * only_c[] = { c };
        CHECK(llama_moe_cache_materialize_mapped(only_c, 1) == nullptr);
        CHECK(c->data == c_data && c->buffer == anon.get());
    }

    ggml_tensor * tensors[] = { a, b, c, a, nullptr }; // duplicates and nullptr are skipped
    ggml_backend_buffer_ptr moved(llama_moe_cache_materialize_mapped(tensors, 5));
    CHECK(moved != nullptr);
    CHECK(ggml_backend_buffer_is_host(moved.get()));
    CHECK(ggml_backend_buffer_get_usage(moved.get()) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(moved.get())), "CPU_Mapped") != 0);

    CHECK(a->buffer == moved.get() && b->buffer == moved.get());
    CHECK(c->buffer == anon.get() && c->data == c_data);

    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(moved.get());
    const size_t    bsz  = ggml_backend_buffer_get_size(moved.get());
    for (ggml_tensor * t : { a, b }) {
        const uint8_t * p = (const uint8_t *) t->data;
        CHECK(p >= base && p + ggml_nbytes(t) <= base + bsz);
        CHECK(((uintptr_t) p) % ggml_backend_buffer_get_alignment(moved.get()) == 0);
    }
    CHECK(memcmp(a->data, content.data() + off_a, ggml_nbytes(a)) == 0);
    CHECK(memcmp(b->data, content.data() + off_b, ggml_nbytes(b)) == 0);
    // the copies do not overlap
    CHECK((const uint8_t *) a->data + ggml_nbytes(a) <= (const uint8_t *) b->data || (const uint8_t *) b->data + ggml_nbytes(b) <= (const uint8_t *) a->data);

    // the source can go away, the tensors keep their bytes
    mapped.reset();
#ifdef TEST_HAVE_MMAP
    munmap(region, size);
#else
    memset(region, 0, size);
#endif
    CHECK(memcmp(a->data, content.data() + off_a, ggml_nbytes(a)) == 0);
    CHECK(memcmp(b->data, content.data() + off_b, ggml_nbytes(b)) == 0);

    // already moved: nothing left to do
    CHECK(llama_moe_cache_materialize_mapped(tensors, 5) == nullptr);

    printf("OK\n");
    return 0;
}
