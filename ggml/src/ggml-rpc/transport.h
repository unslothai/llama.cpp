#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

struct rpc_deferred_op {
    enum kind_t { GET, SET } kind;

    // serialized at queue time: the owning graph result is reset once per ubatch, so a pointer would dangle
    std::vector<uint8_t> tensor_bytes;

    void *   data   = nullptr;   // GET: host destination; SET: host staging source
    uint64_t offset = 0;
    uint64_t size   = 0;

    // SET only: must complete before the staging buffer holds the data. void to keep ggml-backend out.
    void * event = nullptr;
};

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

// One connection is shared by every backend of an endpoint, across llama_contexts, so: mtx_send
// keeps a message atomic on the wire; seq_* hands responses out in request order without holding
// mtx_send; last_graph_uid stops RECOMPUTE re-running another context's graph.
struct rpc_conn_state {
    std::mutex              mtx_send;
    std::mutex              mtx_seq;
    std::condition_variable cv_seq;
    uint64_t                seq_next    = 0;
    uint64_t                seq_serving = 0;

    std::unordered_map<uint32_t, uint64_t> last_graph_uid;

    uint32_t server_minor = 0;

    // lock order: mtx_defer before mtx_send, never the reverse
    std::mutex                   mtx_defer;
    std::vector<rpc_deferred_op> deferred;
};

struct socket_t {
    ~socket_t();

    rpc_conn_state conn;

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
