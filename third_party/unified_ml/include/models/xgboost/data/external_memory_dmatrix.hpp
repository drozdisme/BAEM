#pragma once
//                                        
//  external_memory_dmatrix.h  — Out-of-core / External Memory DMatrix
//
//  Motivation (§4.2, Chen & Guestrin):
//    When the training dataset does not fit in RAM, we cannot load it into a
//    DMatrix directly.  The paper describes a "block-based loading" strategy
//    where the data is partitioned into fixed-size blocks on disk, each block
//    sorted by column for sequential access.
//
//  Design:
//    • Data is stored as binary row-major blocks on disk (one file per block,
//      or a single file with a block index).
//    • ExternalMemoryDMatrix exposes the same feature() interface as DMatrix,
//      loading blocks on demand via buffered IO (or mmap on POSIX systems).
//    • A small LRU cache keeps the K most recently used blocks in memory.
//      When a new block is needed, the least-recently-used block is evicted.
//    • HistogramBuilder and SplitEvaluator iterate blocks sequentially, so
//      effective access is nearly sequential despite external storage.
//
//  File format — BlockFile:
//    Header: [n_rows: uint32][n_cols: uint32]
//    Data:   n_rows × n_cols × float32 (row-major)
//    Labels: n_rows × float32
//
//  Usage:
//    ExternalMemoryDMatrix dm("train.cache", n_cols, cache_blocks=4);
//    dm.add_block_file("block_0.bin");
//    dm.add_block_file("block_1.bin");
//    ...
//    // Then use dm as if it were a DMatrix inside Booster::train()
//
//  Thread safety:
//    load_block() uses a mutex; parallel reads to different blocks are safe.
//    write_block() is NOT thread-safe (call from single-threaded setup only).
//                                        
#include "models/xgboost/core/types.hpp"
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <future>
#include <limits>

namespace xgb {

//                                        
//  BlockMeta  — lightweight descriptor of one on-disk block
//                                        
struct BlockMeta {
    std::string filepath;  // path to the binary block file
    bst_uint    n_rows;    // number of rows in this block
    bst_uint    n_cols;    // number of features (same for all blocks)
    bst_uint    global_row_offset;  // first row index in global space
};

//                                        
//  InMemoryBlock  — a loaded block in the LRU cache
//                                        
struct InMemoryBlock {
    bst_uint block_id;
    std::vector<bst_float> data;    // row-major: data[r * n_cols + c]
    std::vector<bst_float> labels;

    bst_float feature(bst_uint local_row, bst_uint col, bst_uint n_cols) const {
        return data[local_row * n_cols + col];
    }
};

//                                        
//  ExternalMemoryDMatrix  — streaming DMatrix backed by disk blocks
//                                        
class ExternalMemoryDMatrix {
public:
    // cache_size: number of blocks to keep in RAM simultaneously.
    // Larger values reduce IO but increase peak memory.
    explicit ExternalMemoryDMatrix(bst_uint n_cols,
                                   bst_uint cache_size = 4)
        : n_cols_(n_cols), cache_size_(cache_size) {}

    //   Block registration                         

    // Register a pre-existing binary block file
    void add_block_file(const std::string& filepath);

    // Write a dense chunk to a new block file and register it
    static void write_block_file(
        const std::string& filepath,
        const std::vector<bst_float>& data,   // row-major, n_rows × n_cols
        const std::vector<bst_float>& labels,
        bst_uint n_rows,
        bst_uint n_cols);

    //   DMatrix-compatible interface                    

    bst_uint num_rows()     const { return total_rows_; }
    bst_uint num_cols()     const { return n_cols_; }
    bst_uint num_features() const { return n_cols_; }
    bst_uint num_blocks()   const {
        return static_cast<bst_uint>(block_meta_.size());
    }

    // Feature access — triggers block load if not cached
    bst_float feature(bst_uint global_row, bst_uint col) const;

    // Global label access — triggers block load if not cached
    bst_float label(bst_uint global_row) const;

    //   Block iteration API (for HistogramBuilder)             
    // Returns an in-memory block by index, loading from disk if needed.
    // Evicts LRU block if cache is full.
    const InMemoryBlock& get_block(bst_uint block_id) const;

    // Iterator interface for sequential block access:
    //   for (bst_uint b = 0; b < dm.num_blocks(); ++b)
    //     process(dm.get_block(b));
    const BlockMeta& block_meta(bst_uint b) const { return block_meta_[b]; }

    // Prefetch block b+1 into cache while processing block b
    void prefetch(bst_uint block_id) const;

    void print_info() const;

private:
    bst_uint n_cols_;
    bst_uint cache_size_;
    bst_uint total_rows_{0};

    std::vector<BlockMeta> block_meta_;

    // LRU cache: deque of block_ids in recency order (front = most recent)
    mutable std::deque<bst_uint>                          lru_order_;
    mutable std::unordered_map<bst_uint, InMemoryBlock>   cache_;
    mutable std::mutex                                    cache_mutex_;

    // Load a block from disk into cache, evicting LRU if needed
    InMemoryBlock read_block_from_disk(bst_uint block_id) const;
    void load_block_into_cache(bst_uint block_id) const;
    void insert_block_into_cache_locked(InMemoryBlock blk) const;
    void evict_lru() const;   // remove oldest block (no mutex — caller holds it)
    void cleanup_completed_prefetch_locked() const;
    bool has_cached_block_locked(bst_uint block_id) const;

    mutable std::future<void> prefetch_future_;
    mutable bst_uint          prefetched_block_id_{std::numeric_limits<bst_uint>::max()};
    mutable bool              prefetch_inflight_{false};
};

//                                        
//  HistogramBuilder extension for external memory
//
//  Builds gradient histograms by streaming through blocks sequentially.
//  Each block is loaded once, processed, then the next block is loaded.
//  Memory high-water mark: cache_size × block_bytes.
//                                        
class HistogramBuilder;

std::vector<struct FeatureHistogram> build_histograms_external(
    const HistogramBuilder& builder,
    const ExternalMemoryDMatrix& dm,
    const std::vector<GradientPair>& grads,          // indexed by global row
    const std::vector<bst_uint>& row_indices,         // global row set for this node
    const std::vector<bst_uint>& col_indices);

} // namespace xgb
