// Byte-exact SET_TENSOR / GET_TENSOR through one RPC endpoint (plain "host:port" or "tls://<fp>@host:port").
// usage: test-rpc-tls <endpoint> [MiB]   (exit 0 = round trip exact; a refused TLS peer aborts non-zero)
//        test-rpc-tls --two-servers <certA> <keyA> <certB> <keyB> <ca> <allow> <portA> <portB>
//        (two TLS servers in one process, each configured just before it starts; runs until killed)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int two_servers(char ** a) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        fprintf(stderr, "no CPU device\n");
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        if (!ggml_backend_rpc_server_set_tls(a[2 * i], a[2 * i + 1], a[4], a[5])) {
            return 1;
        }
        const std::string endpoint = std::string("127.0.0.1:") + a[6 + i];
        std::thread([endpoint, cpu]() mutable {
            ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 1, 1, &cpu);
        }).detach();
        // the server takes its configuration as it starts; only then configure the next one
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <endpoint> [MiB]\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    if (strcmp(argv[1], "--two-servers") == 0) {
        if (argc < 10) {
            fprintf(stderr, "usage: %s --two-servers certA keyA certB keyB ca allow portA portB\n", argv[0]);
            return 2;
        }
        return two_servers(argv + 2);
    }
    const char * endpoint = argv[1];
    const size_t mib      = argc > 2 ? (size_t) atoi(argv[2]) : 64;
    const int64_t n       = (int64_t) (mib * 1024 * 1024 / sizeof(float));

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    if (!backend) {
        fprintf(stderr, "cannot init %s\n", endpoint);
        return 1;
    }
    ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "cannot allocate %zu MiB on %s\n", mib, endpoint);
        return 1;
    }

    std::vector<uint8_t> src(ggml_nbytes(t)), dst(ggml_nbytes(t));
    std::mt19937 rng(1234);
    for (auto & b : src) {
        b = (uint8_t) rng();
    }
    // one large write, then small writes at odd offsets (many messages, each its own TLS record)
    ggml_backend_tensor_set(t, src.data(), 0, src.size());
    for (size_t off = 1; off < 4096; off += 509) {
        src[off] ^= 0x5a;
        ggml_backend_tensor_set(t, src.data() + off, off, 1);
    }
    ggml_backend_tensor_get(t, dst.data(), 0, dst.size());
    const bool exact = memcmp(src.data(), dst.data(), src.size()) == 0;
    size_t small_ok = 0;
    for (size_t off = 3; off < 2048; off += 211) {
        uint8_t b = 0;
        ggml_backend_tensor_get(t, &b, off, 1);
        small_ok += b == src[off];
    }
    printf("%s: %zu MiB round trip %s, small reads %zu/10\n", endpoint, mib, exact ? "exact" : "MISMATCH", small_ok);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return exact && small_ok == 10 ? 0 : 1;
}
