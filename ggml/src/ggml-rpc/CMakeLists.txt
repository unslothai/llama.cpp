message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                         log.cpp
                         transport.cpp
                        )

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()

# RDMA auto-detection: Linux RoCE/IB via libibverbs, Apple RDMA-over-Thunderbolt via librdma
if (APPLE)
    set(RDMA_LIB_NAME rdma)
    set(RDMA_DESC "Apple RDMA-over-Thunderbolt, UC")
elseif (NOT WIN32)
    set(RDMA_LIB_NAME ibverbs)
    set(RDMA_DESC "auto-detected")
endif()

if (RDMA_LIB_NAME)
    find_library(RDMA_LIB ${RDMA_LIB_NAME})
    if (RDMA_LIB)
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" ON)
    else()
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" OFF)
    endif()
else()
    set(GGML_RPC_RDMA OFF CACHE BOOL "RDMA not available on this platform" FORCE)
endif()

if (GGML_RPC_RDMA)
    if (NOT RDMA_LIB)
        find_library(RDMA_LIB ${RDMA_LIB_NAME} REQUIRED)
    endif()
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
    if (APPLE)
        # librdma.dylib only exists on macOS 26.2 and later. Link it weakly so a build made
        # where it exists still loads where it does not; checked at runtime before use
        # but with BUILD_SHARED_LIBS=OFF ggml-rpc is a static archive and never links
        # so the librdma symbols used by transport-apple.cpp stay undefined.
        target_link_options(ggml-rpc PUBLIC "LINKER:-weak_library,${RDMA_LIB}")
        target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA_APPLE)
        target_sources(ggml-rpc PRIVATE transport-apple.cpp)
    else()
        target_link_libraries(ggml-rpc PRIVATE ${RDMA_LIB})
    endif()
    message(STATUS "  RDMA transport enabled (${RDMA_DESC})")
else()
    message(STATUS "  RDMA transport disabled")
endif()

# Mutual TLS 1.3 transport ("tls://<fingerprint>@host:port" endpoints, rpc-server --tls-*)
option(GGML_RPC_TLS           "ggml: mutual TLS transport for RPC"                     OFF)

if (GGML_RPC_TLS)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(ggml-rpc PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "  TLS transport enabled (OpenSSL ${OPENSSL_VERSION})")
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_TLS)
else()
    message(STATUS "  TLS transport disabled")
endif()
