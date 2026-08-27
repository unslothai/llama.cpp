#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <utility>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    // opt-in, see llama_mmap_random_mode(). marks one byte range as randomly accessed, which is
    // only correct once loading is done: until then the loader streams the file sequentially.
    // offsets are into the file, which is also the offset into the mapping - the whole file is
    // always mapped from zero. the range is rounded out to whole pages, since madvise needs that.
    void advise_random_range(size_t offset, size_t len, bool drop);

    // eager pull-in for everything outside the given ranges, used in place of the constructor's
    // whole-file one when part of the file must not be read ahead. ranges must be sorted.
    void prefetch_except(const std::vector<std::pair<size_t, size_t>> & skip);

    // true if [ptr, ptr + len) lies inside this mapping
    bool contains(const void * ptr, size_t len) const;

    // ask the kernel to start reading the given rows. issued as one batch so the faults overlap
    // instead of serializing.
    void prefetch_rows(const void * base, size_t stride, size_t row_size,
                       const int32_t * rows, size_t n_rows) const;

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// how the model file mappings should be advised, from the LLAMA_MMAP_RANDOM environment variable.
// off unless the user asks: the random hints cost a large cold-prefill slowdown on models whose
// host-resident tensors are read sequentially, so this cannot be a default.
enum llama_mmap_random_mode {
    LLAMA_MMAP_RANDOM_OFF  = 0, // upstream behaviour
    LLAMA_MMAP_RANDOM_ON   = 1, // advise the gather tables random after load, do not pull them in
    LLAMA_MMAP_RANDOM_DROP = 2, // additionally drop what the load pulled in
};

llama_mmap_random_mode llama_mmap_random_mode_get();

// whether batched readahead ahead of a sparse gather is enabled (LLAMA_MMAP_RANDOM_PREFETCH)
bool llama_mmap_random_prefetch_enabled();

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
