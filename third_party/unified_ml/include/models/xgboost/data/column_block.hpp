#pragma once
//                                        
//  column_block.h  — Cache-aware Column Block Storage  (§4.1, Chen & Guestrin)
//
//  Problem:
//    Exact greedy split search reads one column at a time and sorts rows by
//    feature value. On a large dataset this produces random access patterns
//    across the gradient/Hessian array, thrashing the CPU L2 cache.
//
//  Solution — Column Blocks:
//    Store each feature column in a sorted, self-contained block that fits in
//    L2 cache (~256 KB on modern CPUs ≈ 2^16 float entries at 4 B each).
//    The XGBoost paper uses 2^22 entries per block, targeting L3.
//    We default to 2^22 here and let the user tune via COLUMN_BLOCK_SIZE.
//
//    Each block stores a contiguous array of (value, row_idx) pairs sorted
//    ascending by value, for a slice of rows. Scanning a block from left to
//    right gives:
//      • Sequential memory reads (cache lines prefetched automatically)
//      • No random row lookups inside the block
//      • Gradient/Hessian access still random, but amortised across the block
//
//  Architecture:
//    ColumnBlock      — one sorted chunk for one feature × row range
//    ColumnBlockStore — owns all blocks; exposes an iterator interface to
//                       HistogramBuilder and SplitEvaluator.
//
//  Integration:
//    • ColumnBlockStore is built once from DMatrix in Booster::train().
//    • HistogramBuilder::build_histograms_blocked() walks blocks sequentially.
//    • SplitEvaluator can iterate blocks when cfg_.use_column_blocks is true.
//                                        

#include "models/xgboost/core/types.hpp"
#include <vector>
#include <algorithm>
#include <cstddef>

namespace xgb {

//   Tunable: entries per block (targeting L3 cache at 4 B per entry)      
// 2^22 = 4 Mi entries = 32 MiB for a pair<float,uint32> (8 B) — matches the
// value cited in the paper.  Reduce to 2^16 for strict L2 residence.
static constexpr bst_uint kColumnBlockSize = (1u << 22);

//                                        
//  SortedEntry  — value + original row index, sorted ascending by value.
//  8 bytes: aligns naturally; no padding wasted.
//                                        
struct SortedEntry {
    bst_float value;    // feature value  x_{i,f}
    bst_uint  row_idx;  // original row index i

    bool operator<(const SortedEntry& rhs) const {
        return value < rhs.value;
    }
};

//                                        
//  ColumnBlock  — sorted (value, row) pairs for one feature × one row-chunk.
//
//  A feature with n rows is split into ceil(n / kColumnBlockSize) blocks.
//  Each block is contiguous in memory and independently sortable.
//                                        
struct ColumnBlock {
    bst_uint feature_idx;               // which feature
    bst_uint row_begin;                 // first row in this block
    bst_uint row_end;                   // one-past-last row
    std::vector<SortedEntry> entries;   // sorted ascending by value

    bst_uint size() const {
        return static_cast<bst_uint>(entries.size());
    }

    // Pointer to raw sorted array — used for sequential SIMD-friendly reads
    const SortedEntry* data() const { return entries.data(); }
};

//                                        
//  ColumnBlockStore
//
//  Owns all column blocks for all features. Built once from a DMatrix.
//  Layout: blocks_[f] = list of ColumnBlock for feature f (one per row chunk).
//
//  Build cost: O(n_features × n_rows × log(block_size)) — same as one-time
//              column sort in exact greedy.  Amortised across all trees.
//                                        
class ColumnBlockStore {
public:
    ColumnBlockStore() = default;

    // Build all blocks from a dense DMatrix.
    // block_size: max entries per block (default kColumnBlockSize).
    void build(const class DMatrix& dm,
               bst_uint block_size = kColumnBlockSize);

    // Access all blocks for feature f
    const std::vector<ColumnBlock>& blocks_for(bst_uint feature_idx) const {
        return blocks_[feature_idx];
    }

    bst_uint num_features() const {
        return static_cast<bst_uint>(blocks_.size());
    }

    bst_uint num_blocks_for(bst_uint f) const {
        return static_cast<bst_uint>(blocks_[f].size());
    }

    bool empty() const { return blocks_.empty(); }

private:
    // blocks_[feature][block_index]
    std::vector<std::vector<ColumnBlock>> blocks_;
};

} // namespace xgb
