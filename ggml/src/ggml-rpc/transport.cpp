#include "transport.h"
#include "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#  include <sys/uio.h>
#endif
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#ifdef GGML_RPC_TLS
#  include <openssl/err.h>
#  include <openssl/evp.h>
#  include <openssl/ssl.h>
#  include <openssl/x509.h>
#  include <cctype>
#  include <cerrno>
#  include <chrono>
#  include <fstream>
#  ifndef _WIN32
#    include <fcntl.h>
#    include <poll.h>
#    include <pthread.h>
#    include <signal.h>
#  endif
#endif // GGML_RPC_TLS

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <chrono>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
static constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
static constexpr int    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
static constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;   // send completions
    struct ibv_cq * rcq = nullptr;   // recv completions
    struct ibv_qp * qp  = nullptr;
    struct ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    void          * tx_buf = nullptr;
    struct ibv_mr * tx_mr  = nullptr;

    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    struct ibv_mr * rx_mr  = nullptr;
    int             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(int i) const {
        return static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        if (tx_mr) ibv_dereg_mr(tx_mr);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        free(tx_buf);
        free(rx_buf);
        if (qp)  ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (ch)  ibv_destroy_comp_channel(ch);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
struct rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    int      gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;
};

struct rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

struct socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    bool flush();
    void get_caps(uint8_t * local_caps);
    bool update_caps(const uint8_t * remote_caps);

    // TCP: small writes of one message (command byte, size, payload) are held here and leave in a
    // single send at flush(), which every message ends with, instead of one tiny segment each.
    // The bytes on the wire are unchanged, only how they are split into syscalls.
    bool tcp_send(const void * data, size_t size);
    bool tcp_send2(const void * a, size_t na, const void * b, size_t nb);
    bool tcp_flush();
    static constexpr size_t TCP_COALESCE_MAX = 64 * 1024;
    std::vector<uint8_t> pending;

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  else
    bool rdma_probe();
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);
    bool tcp_peer_closed();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc);
    bool rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;

#ifdef GGML_RPC_TLS
    // one TLS record per message: the small writes of a message are held and leave together at flush()
    static constexpr size_t TLS_COALESCE_MAX = 64 * 1024;
    SSL *                ssl = nullptr;
    bool                 tls_broken = false; // a read or write failed: no close_notify on a dead session
    std::vector<uint8_t> tls_out;
    bool tls_write(const void * data, size_t size);
    bool tls_send(const void * data, size_t size);
    bool tls_flush();
    bool tls_recv(void * data, size_t size);
#endif // GGML_RPC_TLS
};

#ifdef GGML_RPC_TLS
static void tls_close_notify(SSL * ssl, sockfd_t fd);
#endif // GGML_RPC_TLS

socket_t::impl::~impl() {
    if (!use_rdma && !pending.empty()) {
        tcp_flush();
    }
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
#ifdef GGML_RPC_TLS
    if (ssl) {
        if (!tls_broken && tls_out.empty()) {
            tls_close_notify(ssl, fd);
        }
        SSL_free(ssl);
        ssl = nullptr;
    }
#endif // GGML_RPC_TLS
    LOG_DBG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
    if (fd != INVALID_SOCKET) closesocket(this->fd);
#else
    if (fd >= 0) close(this->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        return std::nullopt;
    }
    rdma_gid_t target = {};
    if (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        return target;
    }
    if (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        return target;
    }
    return std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

bool socket_t::impl::tcp_peer_closed() {
    if (fd < 0) return false;
#ifndef _WIN32
    struct pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    int r = poll(&pfd, 1, 0);
    return r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#else
    return false;
#endif
}

bool socket_t::impl::rdma_probe() {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    const char * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    if (!target_gid) {
        return false;
    }

    const uint8_t ib_port = 1;
    int num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    if (!devs || num_devs == 0) return false;

    ibv_context * ibctx = nullptr;
    const char * matched_dev = nullptr;
    int gid_idx = gid_env ? atoi(gid_env) : -1;
    int gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    for (int d = 0; d < num_devs; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;

        ibv_port_attr pa;
        if (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); continue; }

        int found_gid = gid_idx;
        int found_version = IBV_GID_TYPE_IB;
        if (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            int v2_idx = -1;
            int v1_idx = -1;
            for (int i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                if (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) continue;
                if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } else if (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            if (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } else if (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } else {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            if (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        if (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            break;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) return false;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    if (!rdma->pd) return false;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    if (!rdma->scq || !rdma->rcq) return false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) return false;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!rdma->tx_buf || !rdma->rx_buf) return false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rdma->tx_mr || !rdma->rx_mr) return false;

    ibv_gid local_gid;
    if (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) return false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const char * ver_str = "";
    if (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } else if (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    return true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    // RESET -> INIT
    {
        struct ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            return false;
        }
    }

    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!rdma->post_rx(i)) return false;
    }

    // INIT -> RTR
    {
        struct ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return false;
        }
    }

    // RTR -> RTS
    {
        struct ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return false;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    return true;
}

// Sleep until the completion channel has an event or the TCP peer closes.
bool socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    struct pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    if (poll(pfds, 2, -1) < 0) {
        return errno == EINTR;
    }
    if (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        return false;
    }
    if (pfds[0].revents & POLLIN) {
        struct ibv_cq * ev_cq  = nullptr;
        void          * ev_ctx = nullptr;
        if (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            return false;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    return true;
}

bool socket_t::impl::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    bool armed = false;
    for (uint64_t s = 1; ; s++) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            if (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            return wc->status == IBV_WC_SUCCESS;
        }
        if (n < 0) return false;
        if (armed) {
            // armed and still empty: sleep until the next completion
            if (!rdma_wait_event()) {
                return false;
            }
            armed = false;
            continue;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        if (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            if (ibv_req_notify_cq(cq, 0) != 0) {
                return false;
            }
            armed = true;
            continue;
        }
        if ((s & 0xFFFFF) == 0) {
            if (tcp_peer_closed()) {
                return false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } else {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;
        c->last_active = std::chrono::steady_clock::now();
        struct ibv_wc wc;
        if (!rdma_poll(c->scq, &wc)) return false;

        src += chunk;
        rem -= chunk;
    }
    return true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        struct ibv_wc wc;
        if (!rdma_poll(c->rcq, &wc)) return false;

        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        if (!c->post_rx(slot)) return false;

        dst += got;
        rem -= got;
    }
    return true;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

bool socket_t::impl::tcp_send(const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

// Send a then b with as few syscalls as possible and without copying b (a large payload).
bool socket_t::impl::tcp_send2(const void * a, size_t na, const void * b, size_t nb) {
    if (nb > MAX_CHUNK_SIZE) {
        return tcp_send(a, na) && tcp_send(b, nb);
    }
#ifdef _WIN32
    WSABUF bufs[2];
    bufs[0].buf = (CHAR *)a;
    bufs[0].len = (ULONG)na;
    bufs[1].buf = (CHAR *)b;
    bufs[1].len = (ULONG)nb;
    DWORD n = 0;
    if (WSASend(fd, bufs, 2, &n, 0, nullptr, nullptr) != 0) {
        LOG_ERROR("send failed (bytes_sent=0, size_to_send=%zu)\n", na + nb);
        return false;
    }
    size_t sent = (size_t)n;
#else
    struct iovec iov[2];
    iov[0].iov_base = const_cast<void *>(a);
    iov[0].iov_len  = na;
    iov[1].iov_base = const_cast<void *>(b);
    iov[1].iov_len  = nb;
    struct msghdr msg = {};
    msg.msg_iov    = iov;
    msg.msg_iovlen = 2;
    ssize_t n = sendmsg(fd, &msg, 0);
    if (n < 0) {
        LOG_ERROR("send failed (bytes_sent=0, size_to_send=%zu)\n", na + nb);
        return false;
    }
    size_t sent = (size_t)n;
#endif
    // a partial send finishes with plain sends of whatever is left of a, then of b
    if (sent < na) {
        return tcp_send((const char *)a + sent, na - sent) && tcp_send(b, nb);
    }
    sent -= na;
    return sent >= nb || tcp_send((const char *)b + sent, nb - sent);
}

bool socket_t::impl::tcp_flush() {
    if (pending.empty()) {
        return true;
    }
    bool ok = tcp_send(pending.data(), pending.size());
    pending.clear();
    return ok;
}

bool socket_t::impl::send_data(const void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_TLS
    if (ssl) {
        return tls_send(data, size);
    }
#endif // GGML_RPC_TLS
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->send(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    if (pending.size() + size <= TCP_COALESCE_MAX) {
        const uint8_t * p = (const uint8_t *)data;
        pending.insert(pending.end(), p, p + size);
        return true;
    }
    if (pending.empty()) {
        return tcp_send(data, size);
    }
    bool ok = tcp_send2(pending.data(), pending.size(), data, size);
    pending.clear();
    return ok;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_TLS
    if (ssl) {
        return tls_recv(data, size);
    }
#endif // GGML_RPC_TLS
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->recv(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    // never wait for a reply while part of our own request is still held back
    if (!pending.empty() && !tcp_flush()) {
        return false;
    }
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (char *)data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            return false;
        }
        bytes_recv += (size_t)n;
    }
    return true;
}

void socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_TLS
    if (ssl) {
        return; // no RDMA under TLS: every byte stays inside the authenticated channel
    }
#endif // GGML_RPC_TLS
#ifdef GGML_RPC_RDMA
    if (std::getenv("GGML_RPC_NO_RDMA")) {
        return;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    if (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    if (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  else
    rdma_local = {};
    if (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
    } else {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

bool socket_t::impl::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_TLS
    if (ssl) {
        (void)remote_caps;
        return tls_flush();
    }
#endif // GGML_RPC_TLS
    // the caps just written go over TCP: send them before the transport may switch to RDMA, and
    // report a failed send, which queued writes only surface here
    if (!tcp_flush()) {
        return false;
    }
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    bool remote_rdma = false;
    for (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    if (!rdma || !remote_rdma) {
        if (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        return true;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    bool activated = rdma->activate(remote_caps);
#  else
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, sizeof(rc));
    bool activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    if (activated) {
        use_rdma = true;
    } else {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#else
    (void)remote_caps;
#endif // GGML_RPC_RDMA
    return true;
}

bool socket_t::impl::flush() {
#ifdef GGML_RPC_TLS
    if (ssl) {
        return tls_flush();
    }
#endif // GGML_RPC_TLS
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->flush();
    }
#endif
    return tcp_flush();
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

bool socket_t::flush() {
    return pimpl->flush();
}

void socket_t::get_caps(uint8_t * local_caps) {
    return pimpl->get_caps(local_caps);
}

bool socket_t::update_caps(const uint8_t * remote_caps) {
    return pimpl->update_caps(remote_caps);
}

const char * socket_t::transport_name() const {
    if (is_tls()) {
        return "TLS";
    }
    return pimpl->use_rdma ? "RDMA" : "TCP";
}

bool socket_t::is_tls() const {
#ifdef GGML_RPC_TLS
    return pimpl->ssl != nullptr;
#else
    return false;
#endif // GGML_RPC_TLS
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    return socket_ptr(new socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (::connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) {
        return true;
    }
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        return false;
    }
    g_rpc_transport_wsa_started = true;
    return true;
#else
    return true;
#endif
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (!g_rpc_transport_wsa_started) {
        return;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#endif
}

/////////////////////////////////////////////////////////////////////////////
// Mutual TLS 1.3. Identity is the SHA-256 fingerprint of the peer's leaf certificate (DER), on top of a
// chain that must verify against the configured CA; host names play no part.

#ifdef GGML_RPC_TLS

static constexpr int    TLS_HANDSHAKE_TIMEOUT_S = 10;
static constexpr size_t TLS_WRITE_CHUNK         = 16u * 1024u * 1024u;

static void tls_log_errors(const char * what) {
    unsigned long e;
    bool any = false;
    while ((e = ERR_get_error()) != 0) {
        char buf[256];
        ERR_error_string_n(e, buf, sizeof(buf));
        LOG_ERROR("%s: %s\n", what, buf);
        any = true;
    }
    if (!any) {
        LOG_ERROR("%s\n", what);
    }
}

static std::string tls_normalize_fp(const std::string & in) {
    std::string out;
    for (char c : in) {
        if (std::isxdigit((unsigned char) c)) {
            out += (char) std::tolower((unsigned char) c);
        }
    }
    return out;
}

static std::string tls_peer_fingerprint(SSL * ssl) {
#if defined(OPENSSL_IS_BORINGSSL) || OPENSSL_VERSION_NUMBER < 0x30000000L
    X509 * cert = SSL_get_peer_certificate(ssl);
#else
    X509 * cert = SSL_get1_peer_certificate(ssl);
#endif
    if (!cert) {
        return "";
    }
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int  n = 0;
    const bool    ok = X509_digest(cert, EVP_sha256(), md, &n) == 1;
    X509_free(cert);
    if (!ok) {
        return "";
    }
    static const char * hex = "0123456789abcdef";
    std::string fp;
    for (unsigned int i = 0; i < n; i++) {
        fp += hex[md[i] >> 4];
        fp += hex[md[i] & 15];
    }
    return fp;
}

static SSL_CTX * tls_ctx_new(bool server, const char * cert, const char * key, const char * ca) {
    SSL_CTX * ctx = SSL_CTX_new(server ? TLS_server_method() : TLS_client_method());
    if (!ctx) {
        tls_log_errors("TLS: cannot create context");
        return nullptr;
    }
    // TLS 1.3 or nothing: a library that cannot pin both bounds would otherwise negotiate older versions
    if (SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) != 1 ||
        SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION) != 1) {
        tls_log_errors("TLS: this TLS library cannot enforce TLS 1.3");
        SSL_CTX_free(ctx);
        return nullptr;
    }
    SSL_CTX_set_options(ctx, SSL_OP_NO_TICKET);
#ifndef OPENSSL_IS_BORINGSSL
    SSL_CTX_set_ciphersuites(ctx, "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384");
    SSL_CTX_set_num_tickets(ctx, 0);
#endif
    if (SSL_CTX_use_certificate_chain_file(ctx, cert) != 1 ||
        SSL_CTX_use_PrivateKey_file(ctx, key, SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_check_private_key(ctx) != 1 ||
        SSL_CTX_load_verify_locations(ctx, ca, nullptr) != 1) {
        tls_log_errors("TLS: cannot load certificate, key or CA");
        SSL_CTX_free(ctx);
        return nullptr;
    }
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
    return ctx;
}

// bound the handshake so a silent peer cannot hold the (single) accept loop
static void tls_set_timeouts(sockfd_t fd, int seconds) {
#ifdef _WIN32
    DWORD ms = (DWORD) seconds * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *) &ms, sizeof(ms));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *) &ms, sizeof(ms));
#else
    struct timeval tv;
    tv.tv_sec  = seconds;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

static bool tls_set_nonblocking(sockfd_t fd, bool on) {
#ifdef _WIN32
    u_long mode = on ? 1 : 0;
    return ioctlsocket(fd, FIONBIO, &mode) == 0;
#else
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }
    return fcntl(fd, F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK)) == 0;
#endif
}

// The whole handshake runs against one wall-clock deadline: a per-receive timeout alone lets a peer
// trickle a fragment before each one expires and hold the (single) accept loop indefinitely.
static bool tls_handshake(SSL * ssl, sockfd_t fd, bool server, int seconds) {
    if (!tls_set_nonblocking(fd, true)) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    bool       ok       = false;
    for (;;) {
        const int r = server ? SSL_accept(ssl) : SSL_connect(ssl);
        if (r == 1) {
            ok = true;
            break;
        }
        const int err = SSL_get_error(ssl, r);
        if (err != SSL_ERROR_WANT_READ && err != SSL_ERROR_WANT_WRITE) {
            break;
        }
        const long long left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            LOG_ERROR("TLS: handshake did not finish within %d s\n", seconds);
            break;
        }
#ifdef _WIN32
        WSAPOLLFD pfd = {};
        pfd.fd        = fd;
        pfd.events    = err == SSL_ERROR_WANT_READ ? POLLRDNORM : POLLWRNORM;
        if (WSAPoll(&pfd, 1, (int) left) < 0) {
            break;
        }
#else
        struct pollfd pfd = {};
        pfd.fd            = fd;
        pfd.events        = err == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT;
        if (poll(&pfd, 1, (int) left) < 0 && errno != EINTR) {
            break;
        }
#endif
    }
    if (!tls_set_nonblocking(fd, false)) {
        return false;
    }
    return ok;
}

// An orderly end of a healthy session: without close_notify the peer sees every disconnect as a
// truncated stream. Bounded (send timeout), and a peer that already reset the connection must not
// stop this process with SIGPIPE.
static void tls_close_notify(SSL * ssl, sockfd_t fd) {
    tls_set_timeouts(fd, 1);
#ifdef _WIN32
    SSL_shutdown(ssl);
#else
    sigset_t pipe_set;
    sigset_t old_set;
    sigset_t pending;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    sigpending(&pending);
    const bool was_pending = sigismember(&pending, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set);
    SSL_shutdown(ssl);
    sigpending(&pending);
    if (!was_pending && sigismember(&pending, SIGPIPE)) {
        int sig = 0;
        sigwait(&pipe_set, &sig); // consume the SIGPIPE this write raised
    }
    pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
#endif
    ERR_clear_error();
}

// One listening server's certificate, CA and allowlist. rpc_tls_server_configure() prepares the
// configuration for the server started next, which takes it: a later configure never changes a
// server that is already listening.
struct rpc_tls_server_config {
    SSL_CTX *   ctx = nullptr;
    std::string allow_file;
    ~rpc_tls_server_config() {
        if (ctx) {
            SSL_CTX_free(ctx);
        }
    }
};

struct tls_pending_state {
    std::mutex                             mu;
    std::shared_ptr<rpc_tls_server_config> next;
};

static tls_pending_state & tls_pending() {
    static tls_pending_state st;
    return st;
}

static bool tls_allowed(const std::string & allow_file, const std::string & fp) {
    std::ifstream in(allow_file);
    if (!in) {
        LOG_ERROR("TLS: cannot read allowlist %s\n", allow_file.c_str());
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }
        const std::string entry = tls_normalize_fp(line);
        if (!entry.empty() && entry == fp) {
            return true;
        }
    }
    return false;
}

bool socket_t::impl::tls_write(const void * data, size_t size) {
    const char * p = (const char *) data;
    while (size > 0) {
        const int n = SSL_write(ssl, p, (int) std::min(size, TLS_WRITE_CHUNK));
        if (n <= 0) {
            tls_broken = true;
            tls_log_errors("TLS: write failed");
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

bool socket_t::impl::tls_send(const void * data, size_t size) {
    if (tls_out.size() + size <= TLS_COALESCE_MAX) {
        const uint8_t * p = (const uint8_t *) data;
        tls_out.insert(tls_out.end(), p, p + size);
        return true;
    }
    return tls_flush() && tls_write(data, size);
}

bool socket_t::impl::tls_flush() {
    if (tls_out.empty()) {
        return true;
    }
    const bool ok = tls_write(tls_out.data(), tls_out.size());
    tls_out.clear();
    return ok;
}

bool socket_t::impl::tls_recv(void * data, size_t size) {
    // never wait for a reply while part of our own request is still held back
    if (!tls_flush()) {
        return false;
    }
    char * p = (char *) data;
    while (size > 0) {
        const int n = SSL_read(ssl, p, (int) std::min(size, TLS_WRITE_CHUNK));
        if (n <= 0) {
            tls_broken = true;
            if (SSL_get_error(ssl, n) == SSL_ERROR_ZERO_RETURN) {
                LOG_WARN("TLS: peer closed the connection\n");
            } else {
                tls_log_errors("TLS: read failed");
            }
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

bool socket_t::tls_connect(const std::string & server_fingerprint) {
    const char * cert = std::getenv("GGML_RPC_TLS_CERT");
    const char * key  = std::getenv("GGML_RPC_TLS_KEY");
    const char * ca   = std::getenv("GGML_RPC_TLS_CA");
    if (!cert || !key || !ca) {
        LOG_ERROR("TLS: a tls:// endpoint needs GGML_RPC_TLS_CERT, GGML_RPC_TLS_KEY and GGML_RPC_TLS_CA\n");
        return false;
    }
    const std::string want = tls_normalize_fp(server_fingerprint);
    if (want.size() != 64) {
        LOG_ERROR("TLS: the endpoint fingerprint must be a SHA-256 hex digest (64 hex digits)\n");
        return false;
    }
    SSL_CTX * ctx = tls_ctx_new(false, cert, key, ca);
    if (!ctx) {
        return false;
    }
    SSL * ssl = SSL_new(ctx);
    SSL_CTX_free(ctx); // the SSL keeps its own reference
    if (!ssl || SSL_set_fd(ssl, (int) pimpl->fd) != 1) {
        tls_log_errors("TLS: cannot create session");
        SSL_free(ssl);
        return false;
    }
    if (!tls_handshake(ssl, pimpl->fd, false, TLS_HANDSHAKE_TIMEOUT_S)) {
        tls_log_errors("TLS: handshake with the server failed");
        SSL_free(ssl);
        return false;
    }
    const std::string got = tls_peer_fingerprint(ssl);
    if (got != want) {
        LOG_ERROR("TLS: server certificate %s is not the pinned %s\n", got.c_str(), want.c_str());
        SSL_free(ssl);
        return false;
    }
    pimpl->ssl = ssl;
    return true;
}

bool socket_t::tls_accept(const rpc_tls_server_config & config, std::string & client_fingerprint) {
    if (!config.ctx) {
        return false;
    }
    SSL *               ssl        = SSL_new(config.ctx);
    const std::string & allow_file = config.allow_file;
    if (!ssl || SSL_set_fd(ssl, (int) pimpl->fd) != 1) {
        tls_log_errors("TLS: cannot create session");
        SSL_free(ssl);
        return false;
    }
    if (!tls_handshake(ssl, pimpl->fd, true, TLS_HANDSHAKE_TIMEOUT_S)) {
        tls_log_errors("TLS: handshake with the client failed");
        SSL_free(ssl);
        return false;
    }
    client_fingerprint = tls_peer_fingerprint(ssl);
    if (client_fingerprint.empty() || !tls_allowed(allow_file, client_fingerprint)) {
        LOG_WARN("TLS: client certificate %s is not in the allowlist\n", client_fingerprint.c_str());
        SSL_free(ssl);
        return false;
    }
    pimpl->ssl = ssl;
    return true;
}

bool rpc_tls_supported() {
    return true;
}

bool rpc_tls_server_configure(const char * cert, const char * key, const char * ca, const char * allow_file) {
    if (!cert || !key || !ca || !allow_file) {
        LOG_ERROR("TLS: the server needs a certificate, a key, a CA and an allowlist\n");
        return false;
    }
    SSL_CTX * ctx = tls_ctx_new(true, cert, key, ca);
    if (!ctx) {
        return false;
    }
    if (!std::ifstream(allow_file)) {
        LOG_ERROR("TLS: cannot read allowlist %s\n", allow_file);
        SSL_CTX_free(ctx);
        return false;
    }
    auto config        = std::make_shared<rpc_tls_server_config>();
    config->ctx        = ctx;
    config->allow_file = allow_file;
    tls_pending_state &         st = tls_pending();
    std::lock_guard<std::mutex> lock(st.mu);
    st.next = std::move(config);
    return true;
}

std::shared_ptr<rpc_tls_server_config> rpc_tls_server_take() {
    tls_pending_state &         st = tls_pending();
    std::lock_guard<std::mutex> lock(st.mu);
    return std::move(st.next);
}

#else // !GGML_RPC_TLS

bool socket_t::tls_connect(const std::string &) {
    LOG_ERROR("TLS: this build has no TLS support (GGML_RPC_TLS=OFF)\n");
    return false;
}

struct rpc_tls_server_config {};

bool socket_t::tls_accept(const rpc_tls_server_config &, std::string &) {
    return false;
}

bool rpc_tls_supported() {
    return false;
}

bool rpc_tls_server_configure(const char *, const char *, const char *, const char *) {
    LOG_ERROR("TLS: this build has no TLS support (GGML_RPC_TLS=OFF)\n");
    return false;
}

std::shared_ptr<rpc_tls_server_config> rpc_tls_server_take() {
    return nullptr;
}

#endif // GGML_RPC_TLS
