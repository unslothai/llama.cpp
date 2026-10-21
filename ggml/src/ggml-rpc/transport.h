#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);
    const char * transport_name() const;

    // Mutual TLS 1.3 on an already connected TCP socket (GGML_RPC_TLS builds; false otherwise).
    // Client: verifies the server chain against GGML_RPC_TLS_CA and pins its leaf SHA-256 fingerprint,
    // presenting GGML_RPC_TLS_CERT / GGML_RPC_TLS_KEY. Server: uses the rpc_tls_server_configure() setup
    // and admits only client leaves whose fingerprint is in the allowlist file, re-read per connection.
    bool tls_connect(const std::string & server_fingerprint);
    bool tls_accept(std::string & client_fingerprint);
    bool is_tls() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();

bool rpc_tls_supported();
// load the server certificate, key and CA once; the allowlist path is re-read on every accept
bool rpc_tls_server_configure(const char * cert, const char * key, const char * ca, const char * allow_file);
bool rpc_tls_server_enabled();
