// llama_moe_cache_available_memory_linux: the memory the MoE cache may use to copy memory-mapped experts, read from a fake /proc and
// /sys/fs/cgroup: the limits of the ancestor cgroups and of cgroup v1 apply, tmpfs/shmem is not reclaimable, exhausted is not unknown

#include "../src/llama-ext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

namespace fs = std::filesystem;

static const uint64_t GiB = UINT64_C(1) << 30;

static void put(const fs::path & root, const std::string & rel, const std::string & content) {
    const fs::path p = root / rel;
    fs::create_directories(p.parent_path());
    std::ofstream f(p);
    f << content;
}

static std::string n(uint64_t v) {
    return std::to_string(v);
}

static uint64_t avail(const fs::path & root) {
    const uint64_t res = llama_moe_cache_available_memory_linux(root.string().c_str());
    fprintf(stderr, "%-24s %.3f GiB\n", root.filename().string().c_str(), res/(double) GiB);
    return res;
}

int main() {
    const fs::path base = fs::current_path() / "test-moe-cache-available-memory.tmp";
    fs::remove_all(base);
    const std::string meminfo = "MemTotal:       134217728 kB\nMemFree:        1024 kB\nMemAvailable:   " + n(64*GiB/1024) + " kB\n";

    // no files at all: unknown
    CHECK(avail(base / "unknown") == 0);

    // no cgroup limit: MemAvailable
    {
        const fs::path r = base / "unlimited";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "0::/\n");
        CHECK(avail(r) == 64*GiB);
    }
    // healthy v2 leaf limit
    {
        const fs::path r = base / "v2_leaf";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "0::/a/b\n");
        put(r, "sys/fs/cgroup/a/memory.max", "max\n");
        put(r, "sys/fs/cgroup/a/b/memory.max", n(32*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/b/memory.current", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/b/memory.stat", "anon 8589934592\nfile 0\nshmem 0\n");
        CHECK(avail(r) == 24*GiB);
    }
    // unlimited v2 leaf, the parent has 4 GiB left
    {
        const fs::path r = base / "v2_ancestor";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "0::/a/b\n");
        put(r, "sys/fs/cgroup/a/memory.max", n(16*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.current", n(12*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.stat", "anon 12884901888\nfile 0\nshmem 0\n");
        put(r, "sys/fs/cgroup/a/b/memory.max", "max\n");
        put(r, "sys/fs/cgroup/a/b/memory.current", n(12*GiB) + "\n");
        CHECK(avail(r) == 4*GiB);
    }
    // cgroup v1 limit with 4 GiB left (the v1 root reports "unlimited" as a huge number), tmpfs in the cache is not reclaimable
    {
        const fs::path r = base / "v1";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "12:cpu,cpuacct:/docker/x\n5:memory:/docker/x\n0::/\n");
        put(r, "sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712\n");
        put(r, "sys/fs/cgroup/memory/memory.usage_in_bytes", n(20*GiB) + "\n");
        put(r, "sys/fs/cgroup/memory/docker/x/memory.limit_in_bytes", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/memory/docker/x/memory.usage_in_bytes", n(4*GiB) + "\n");
        put(r, "sys/fs/cgroup/memory/docker/x/memory.stat", "cache 1073741824\nshmem 1073741824\ntotal_cache 1073741824\ntotal_shmem 1073741824\n");
        CHECK(avail(r) == 4*GiB);
    }
    // cgroup v1 page cache is reclaimable
    {
        const fs::path r = base / "v1_cache";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "5:memory:/docker/x\n");
        put(r, "sys/fs/cgroup/memory/docker/x/memory.limit_in_bytes", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/memory/docker/x/memory.usage_in_bytes", n(7*GiB) + "\n");
        put(r, "sys/fs/cgroup/memory/docker/x/memory.stat", "cache 3221225472\nshmem 0\ntotal_cache 3221225472\ntotal_shmem 1073741824\n");
        CHECK(avail(r) == 3*GiB);
    }
    // exhausted v2 cgroup: known, not 0 (0 means unknown and lets the copy go ahead)
    {
        const fs::path r = base / "v2_exhausted";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "0::/a\n");
        put(r, "sys/fs/cgroup/a/memory.max", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.current", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.stat", "anon 8589934592\nfile 0\nshmem 0\n");
        CHECK(avail(r) == 1);
    }
    // v2 cgroup whose page cache is mostly tmpfs: only the non-shmem part is reclaimable
    {
        const fs::path r = base / "v2_tmpfs";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "0::/a\n");
        put(r, "sys/fs/cgroup/a/memory.max", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.current", n(7*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.stat", "anon 1073741824\nfile 6442450944\nshmem 6442450944\n");
        CHECK(avail(r) == 1*GiB);
    }
    // v2 cgroup whose page cache is the model file: reclaimable
    {
        const fs::path r = base / "v2_pagecache";
        put(r, "proc/meminfo", meminfo);
        put(r, "proc/self/cgroup", "0::/a\n");
        put(r, "sys/fs/cgroup/a/memory.max", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.current", n(7*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.stat", "anon 1073741824\nfile 6442450944\nshmem 0\n");
        CHECK(avail(r) == 7*GiB);
    }
    // MemAvailable is lower than the cgroup headroom
    {
        const fs::path r = base / "meminfo_lower";
        put(r, "proc/meminfo", "MemAvailable:   " + n(2*GiB/1024) + " kB\n");
        put(r, "proc/self/cgroup", "0::/a\n");
        put(r, "sys/fs/cgroup/a/memory.max", n(8*GiB) + "\n");
        put(r, "sys/fs/cgroup/a/memory.current", n(1*GiB) + "\n");
        CHECK(avail(r) == 2*GiB);
    }

    fs::remove_all(base);
    printf("OK\n");
    return 0;
}
