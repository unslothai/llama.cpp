// [TAG_EXACT_CONCURRENCY] an RPC server must refuse a FLASH_ATTN_EXT with a page table (src[5]), since a client can send any graph.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool port_open(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok = connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0;
    close(fd);
    return ok;
}

[[noreturn]] static void send_graph(const std::string & endpoint, bool paged) {
    alarm(120);

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint.c_str(), 0);
    if (!backend) {
        _exit(10);
    }

    ggml_init_params params = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    const int64_t d = 64, n_kv = 256, n_tokens = 1;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_tokens, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv,     1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv,     1, 1);
    ggml_tensor * pages = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_tokens);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 0.125f, 0.0f, 0.0f);
    if (paged) {
        out->src[5] = pages;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        _exit(11);
    }
    ggml_backend_buffer_clear(buf, 0);

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    _exit(0);
}

static int run_client(const std::string & endpoint, bool paged) {
    const pid_t pid = fork();
    if (pid == 0) {
        send_graph(endpoint, paged);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
}

int main() {
    const int port = 41000 + getpid() % 9000;
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);

    const pid_t server = fork();
    if (server == 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 1, 1, &dev);
        _exit(1);
    }

    bool up = false;
    for (int i = 0; i < 300 && !up; ++i) {
        up = port_open(port);
        if (!up) {
            usleep(100*1000);
        }
    }

    int rc = 0;

    if (!up) {
        fprintf(stderr, "the RPC server did not start on %s\n", endpoint.c_str());
        rc = 1;
    } else {
        const int plain = run_client(endpoint, false);
        const int paged = run_client(endpoint, true);
        const int after = run_client(endpoint, false);

        printf("unpaged: %d, paged: %d, unpaged after: %d\n", plain, paged, after);

        if (plain != 0 || after != 0) {
            fprintf(stderr, "an unpaged graph failed over RPC, so this test proves nothing\n");
            rc = 1;
        } else if (paged == 0) {
            fprintf(stderr, "the RPC server computed a FLASH_ATTN_EXT that carries a page table\n");
            rc = 1;
        } else {
            printf("ok, the RPC server refused the paged graph and kept serving\n");
        }
    }

    kill(server, SIGKILL);
    waitpid(server, nullptr, 0);

    return rc;
}
