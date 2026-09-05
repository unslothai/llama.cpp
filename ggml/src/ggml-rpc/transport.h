#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

// State shared by every client thread that uses one connection. A connection is looked up by
// endpoint and is therefore shared by all backends of that endpoint, including the backends of
// different llama_contexts, so all of it has to be serialised:
//   - mtx_send makes a whole RPC message atomic on the wire
//   - seq_* hands the responses out in request order (the server answers strictly in order),
//     without holding mtx_send while waiting, so another thread can keep submitting work
//   - last_graph_uid mirrors the server's per-connection stored graph for a device, so that
//     RPC_CMD_GRAPH_RECOMPUTE can never re-run a graph submitted by another context
struct rpc_conn_state {
    std::mutex              mtx_send;
    std::mutex              mtx_seq;
    std::condition_variable cv_seq;
    uint64_t                seq_next    = 0;
    uint64_t                seq_serving = 0;

    std::unordered_map<uint32_t, uint64_t> last_graph_uid;
};

struct socket_t {
    ~socket_t();

    // guarded by conn.mtx_send / conn.mtx_seq, see rpc_conn_state
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
