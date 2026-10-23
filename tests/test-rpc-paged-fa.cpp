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
#include <fstream>
#include <sstream>
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

// client exit codes: setup failures are not the server's answer, so they never count as a refusal
enum {
    CLIENT_OK          = 0,
    CLIENT_NO_BACKEND  = 10,
    CLIENT_NO_BUFFER   = 11,
    CLIENT_REFUSED     = 20,  // the server answered the compute with a failure
};

static void on_server_dropped(int) {
    _exit(CLIENT_REFUSED);
}

[[noreturn]] static void send_graph(const std::string & endpoint, bool paged) {
    alarm(120);

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint.c_str(), 0);
    if (!backend) {
        _exit(CLIENT_NO_BACKEND);
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
        _exit(CLIENT_NO_BUFFER);
    }
    ggml_backend_buffer_clear(buf, 0);

    // everything above is setup; from here a failure or a dropped connection is the server's answer.
    // The client aborts (or takes SIGPIPE) when the server closes on it, so only these two, armed here, mean a refusal.
    signal(SIGABRT, on_server_dropped);
    signal(SIGPIPE, on_server_dropped);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        _exit(CLIENT_REFUSED);
    }

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    _exit(0);
}

// a client that could not be started: distinct from every exit status and signal, so it is never read as a refusal
static const int CLIENT_NOT_RUN = -1000;

static int run_client(const std::string & endpoint, bool paged) {
    const pid_t pid = fork();
    if (pid < 0) {
        perror("fork the RPC client");
        return CLIENT_NOT_RUN;
    }
    if (pid == 0) {
        send_graph(endpoint, paged);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
}

// a port the kernel just handed out, so a parallel run or another listener does not answer for the server
static int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_port        = 0;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(addr);
    int port = -1;
    if (bind(fd, (sockaddr *) &addr, sizeof(addr)) == 0 && getsockname(fd, (sockaddr *) &addr, &len) == 0) {
        port = ntohs(addr.sin_port);
    }
    close(fd);
    return port;
}

int main() {
    const int port = free_port();
    if (port <= 0) {
        perror("find a free port");
        return 1;
    }
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);

    // the server's stderr, so the test can see the server itself refuse the graph: a client that fails locally looks the same from the client side
    const char * tmp = getenv("TMPDIR");
    std::string log_path = std::string(tmp && *tmp ? tmp : "/tmp") + "/test-rpc-paged-fa.XXXXXX";
    const int log_fd = mkstemp(&log_path[0]);
    if (log_fd < 0) {
        perror("create a file for the RPC server's log");
        return 1;
    }

    const pid_t server = fork();
    if (server < 0) {
        perror("fork the RPC server");
        close(log_fd);
        unlink(log_path.c_str());
        return 1;  // nothing to kill: kill(-1) would signal every process this user owns
    }
    if (server == 0) {
        dup2(log_fd, STDERR_FILENO);
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

        if (plain == CLIENT_NOT_RUN || paged == CLIENT_NOT_RUN || after == CLIENT_NOT_RUN) {
            fprintf(stderr, "a client could not be started, so this test proves nothing\n");
            rc = 1;
        } else if (plain != 0 || after != 0) {
            fprintf(stderr, "an unpaged graph failed over RPC, so this test proves nothing\n");
            rc = 1;
        } else if (paged == 0) {
            fprintf(stderr, "the RPC server computed a FLASH_ATTN_EXT that carries a page table\n");
            rc = 1;
        } else if (paged != CLIENT_REFUSED) {
            // a setup failure, a signal before the compute, or the client's own timeout: not the server's answer
            fprintf(stderr, "the paged client ended with %d, not a refusal from the server, so this test proves nothing\n", paged);
            rc = 1;
        } else {
            std::ifstream     log_in(log_path);
            std::stringstream log_text;
            log_text << log_in.rdbuf();

            if (log_text.str().find("is a paged FLASH_ATTN_EXT, which RPC does not support") == std::string::npos) {
                fprintf(stderr, "the paged client was refused, but the RPC server did not log refusing the graph: the failure was the client's own\n");
                rc = 1;
            } else {
                printf("ok, the RPC server refused the paged graph and kept serving\n");
            }
        }
    }

    kill(server, SIGKILL);
    waitpid(server, nullptr, 0);

    close(log_fd);
    unlink(log_path.c_str());

    return rc;
}
