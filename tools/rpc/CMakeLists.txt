set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS AND UNIX AND NOT GGML_BACKEND_DL)
    add_executable(test-rpc-multi-server ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.cpp)
    target_link_libraries(test-rpc-multi-server PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-multi-server PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(
        NAME test-rpc-multi-server
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-multi-server>)
    set_property(TEST test-rpc-multi-server PROPERTY LABELS main)
endif()

# the script uses GNU coreutils (timeout, sha256sum, seq), which macOS does not ship
if (LLAMA_BUILD_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT GGML_BACKEND_DL AND GGML_RPC_TLS)
    add_executable(test-rpc-tls ${PROJECT_SOURCE_DIR}/tests/test-rpc-tls.cpp)
    target_link_libraries(test-rpc-tls PRIVATE ggml ggml-rpc)
    add_test(
        NAME test-rpc-tls
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-tls.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-tls>)
    set_property(TEST test-rpc-tls PROPERTY LABELS main)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
