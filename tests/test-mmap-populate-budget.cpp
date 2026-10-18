// llama_mmap::cgroup_budget(): the populate budget clamped to the memory limits of the process's cgroup, on fixture trees

#include "../src/llama-mmap.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static const size_t GiB  = 1024ull*1024*1024;
static const size_t HOST = 64*GiB; // the budget from /proc/meminfo

static int n_fail = 0;

static void put(const fs::path & p, const std::string & s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << s;
}

static void check(const char * name, size_t got, size_t want) {
    const bool ok = got == want;
    printf("%s %-40s got %zu want %zu\n", ok ? "PASS" : "FAIL", name, got, want);
    n_fail += !ok;
}

int main() {
#ifndef __linux__
    printf("SKIP: Linux only\n");
    return 0;
#else
    const fs::path base = fs::absolute("test-mmap-populate-budget.d");
    fs::remove_all(base);

    auto run = [&](const char * name, const std::string & self, size_t want) {
        const fs::path dir = base / name;
        put(dir / "self", self);
        check(name, llama_mmap::cgroup_budget(HOST, (dir / "self").c_str(), (dir / "root").c_str()), want);
    };
    auto v2 = [&](const char * name, const std::string & cg, const char * max, const char * high, const char * cur) {
        const fs::path d = base / name / "root" / cg;
        if (max)  put(d / "memory.max",     max);
        if (high) put(d / "memory.high",    high);
        if (cur)  put(d / "memory.current", cur);
    };
    auto v1 = [&](const char * name, const std::string & cg, const std::string & lim, const std::string & use) {
        const fs::path d = base / name / "root" / "memory" / cg;
        put(d / "memory.limit_in_bytes", lim);
        put(d / "memory.usage_in_bytes", use);
    };
    const std::string G4 = std::to_string(4*GiB), G8 = std::to_string(8*GiB), G1 = std::to_string(1*GiB);
    const std::string G32 = std::to_string(32*GiB);

    // v2: leaf memory.max = 4 GiB, 1 GiB used, ancestors at "max": 4 - 1 - max(2, 4/16) = 1 GiB
    put(base / "v2_leaf" / "root" / "cgroup.controllers", "memory\n");
    v2("v2_leaf", "a/b/c", (G4 + "\n").c_str(), "max\n", (G1 + "\n").c_str());
    v2("v2_leaf", "a/b",   "max\n", "max\n", G1.c_str());
    v2("v2_leaf", "a",     "max\n", "max\n", G1.c_str());
    run("v2_leaf", "0::/a/b/c\n", 1*GiB);

    // v2: memory.high (4 GiB) below memory.max (8 GiB)
    v2("v2_high", "a/b", G8.c_str(), G4.c_str(), G1.c_str());
    v2("v2_high", "a",   "max", "max", G1.c_str());
    run("v2_high", "0::/a/b\n", 1*GiB);

    // v2: the ancestor (4 GiB) is tighter than the leaf (32 GiB)
    v2("v2_ancestor", "a/b", G32.c_str(), "max", G1.c_str());
    v2("v2_ancestor", "a",   G4.c_str(),  "max", G1.c_str());
    run("v2_ancestor", "0::/a/b\n", 1*GiB);

    // v2: 32 GiB, 4 GiB used: 32 - 4 - max(2, 2) = 26 GiB
    v2("v2_large", "a", G32.c_str(), "max", G4.c_str());
    run("v2_large", "0::/a\n", 26*GiB);

    // v2: usage at the limit: nothing to populate
    v2("v2_full", "a", G4.c_str(), "max", G4.c_str());
    run("v2_full", "0::/a\n", 0);

    // v2: "max" everywhere: the host budget
    v2("v2_max", "a/b", "max", "max", G1.c_str());
    v2("v2_max", "a",   "max", "max", G1.c_str());
    run("v2_max", "0::/a/b\n", HOST);

    // v2 in a container (cgroupns): "0::/", the limit on the mount root
    v2("v2_ns", "", G4.c_str(), "max", G1.c_str());
    run("v2_ns", "0::/\n", 1*GiB);

    // v2 without cgroupns: the host path does not exist in the container, its cgroup is the mount root
    v2("v2_bind", "", G4.c_str(), "max", G1.c_str());
    run("v2_bind", "0::/system.slice/docker-1234.scope\n", 1*GiB);

    // v1: memory.limit_in_bytes = 4 GiB, 1 GiB used, other controllers listed first
    v1("v1_leaf", "docker/x", G4, G1);
    v1("v1_leaf", "docker", "9223372036854771712", G1);
    v1("v1_leaf", "", "9223372036854771712", G1);
    run("v1_leaf", "12:cpuset:/docker/x\n4:cpu,cpuacct:/docker/x\n3:memory:/docker/x\n0::/docker/x\n", 1*GiB);

    // v1: unlimited (9223372036854771712): the host budget
    v1("v1_unlimited", "docker/x", "9223372036854771712\n", G1 + "\n");
    run("v1_unlimited", "3:memory:/docker/x\n", HOST);

    // no cgroup info at all: the host budget
    run("none_missing", "", HOST);
    check("none_no_self_file", llama_mmap::cgroup_budget(HOST, (base / "nope").c_str(), (base / "nope_root").c_str()), HOST);

    // controllers file present but no memory files (v1 host without the memory hierarchy, v2 root): the host budget
    put(base / "no_files" / "root" / "cgroup.controllers", "cpu io\n");
    run("no_files", "0::/user.slice\n", HOST);

    // garbage files: ignored, the host budget
    v2("garbage", "a", "lots\n", "", "-\n");
    run("garbage", "0::/a\n", HOST);
    v2("garbage_cur", "a", G4.c_str(), "max", "nope");
    run("garbage_cur", "0::/a\n", HOST);
    run("garbage_self", "this is not a cgroup file\n:\n::\n", HOST);

    // a path leaving the namespace ("/.."): ignored, the host budget
    v2("dotdot", "", G4.c_str(), "max", G1.c_str());
    run("dotdot", "0::/../../x\n", HOST);

    // an unreadable limit file (skipped when running as root, who can read it anyway)
    v2("unreadable", "a", G4.c_str(), "max", G1.c_str());
    fs::permissions(base / "unreadable" / "root" / "a" / "memory.max", fs::perms::none);
    if (std::ifstream(base / "unreadable" / "root" / "a" / "memory.max")) {
        printf("SKIP unreadable (running as root)\n");
    } else {
        run("unreadable", "0::/a\n", HOST);
    }
    fs::permissions(base / "unreadable" / "root" / "a" / "memory.max", fs::perms::owner_all);

    // the budget is only lowered
    v2("lower", "a", G32.c_str(), "max", G1.c_str());
    put(base / "lower" / "self", "0::/a\n");
    check("lower_only", llama_mmap::cgroup_budget(1*GiB, (base / "lower" / "self").c_str(), (base / "lower" / "root").c_str()), 1*GiB);

    fs::remove_all(base);
    printf("%s\n", n_fail ? "FAILED" : "ALL PASS");
    return n_fail ? 1 : 0;
#endif
}
