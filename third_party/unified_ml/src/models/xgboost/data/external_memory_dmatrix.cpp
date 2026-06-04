//              
//  external_memory_dmatrix.cpp  — ExternalMemoryDMatrix implementation
//
//  IO strategy:
//  • Block files use a trivial binary format (header + raw float32 arrays).
//  • Reads use std::ifstream with read() — buffered by the OS page cache.
//  • On POSIX, madvise(MADV_SEQUENTIAL) would further improve prefetching,
//  but we avoid POSIX-only APIs here for cross-platform compatibility.
//  • Prefetch() uses std::async with at most one in-flight block to avoid
//  IO storms while still overlapping compute and data loading.
//              
#include "models/xgboost/data/external_memory_dmatrix.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <chrono>

namespace xgb {

// Binary block file layout:
// [0..3] n_rows  (uint32_t, little-endian)
// [4..7] n_cols  (uint32_t, little-endian)
// [8..]  n_rows × n_cols × float32  (row-major data)
// [8 + n_rows*n_cols*4 ..]  n_rows × float32 (labels)

// write_block_file          
void ExternalMemoryDMatrix::write_block_file(
  const std::string& filepath,
  const std::vector<bst_float>& data,
  const std::vector<bst_float>& labels,
  bst_uint n_rows,
  bst_uint n_cols)
{
  assert(data.size() == static_cast<size_t>(n_rows) * n_cols);
  assert(labels.size() == static_cast<size_t>(n_rows));

  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs) throw std::runtime_error("Cannot write: " + filepath);

  ofs.write(reinterpret_cast<const char*>(&n_rows), sizeof(n_rows));
  ofs.write(reinterpret_cast<const char*>(&n_cols), sizeof(n_cols));
  ofs.write(reinterpret_cast<const char*>(data.data()),
      static_cast<std::streamsize>(data.size() * sizeof(bst_float)));
  ofs.write(reinterpret_cast<const char*>(labels.data()),
      static_cast<std::streamsize>(labels.size() * sizeof(bst_float)));
}

// add_block_file          
void ExternalMemoryDMatrix::add_block_file(const std::string& filepath) {
  // Peek at the header to read n_rows and n_cols
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs) throw std::runtime_error("Cannot open block: " + filepath);

  bst_uint n_rows = 0, n_cols = 0;
  ifs.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
  ifs.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

  if (n_cols != n_cols_ && n_cols_ != 0)
    throw std::runtime_error("Block column count mismatch: expected "
    + std::to_string(n_cols_) + " got " + std::to_string(n_cols));
  n_cols_ = n_cols;

  BlockMeta meta;
  meta.filepath    = filepath;
  meta.n_rows    = n_rows;
  meta.n_cols    = n_cols;
  meta.global_row_offset = total_rows_;
  block_meta_.push_back(meta);
  total_rows_ += n_rows;
}

InMemoryBlock ExternalMemoryDMatrix::read_block_from_disk(bst_uint block_id) const {
  const BlockMeta& meta = block_meta_[block_id];

  std::ifstream ifs(meta.filepath, std::ios::binary);
  if (!ifs) throw std::runtime_error("Cannot open: " + meta.filepath);

  // Skip header (already read during add_block_file)
  bst_uint dummy_rows, dummy_cols;
  ifs.read(reinterpret_cast<char*>(&dummy_rows), sizeof(dummy_rows));
  ifs.read(reinterpret_cast<char*>(&dummy_cols), sizeof(dummy_cols));

  InMemoryBlock blk;
  blk.block_id = block_id;

  // Read feature data
  const size_t data_elems = static_cast<size_t>(meta.n_rows) * meta.n_cols;
  blk.data.resize(data_elems);
  ifs.read(reinterpret_cast<char*>(blk.data.data()),
     static_cast<std::streamsize>(data_elems * sizeof(bst_float)));

  // Read labels
  blk.labels.resize(meta.n_rows);
  ifs.read(reinterpret_cast<char*>(blk.labels.data()),
     static_cast<std::streamsize>(meta.n_rows * sizeof(bst_float)));

  return blk;
}

void ExternalMemoryDMatrix::insert_block_into_cache_locked(InMemoryBlock blk) const {
  const bst_uint block_id = blk.block_id;
  // Evict LRU if cache is full
  if (cache_.size() >= cache_size_) evict_lru();

  // Insert into cache and record recency
  cache_.emplace(block_id, std::move(blk));
  lru_order_.push_front(block_id);
}

// load_block_into_cache
// Reads one block file from disk and inserts it into the LRU cache.
void ExternalMemoryDMatrix::load_block_into_cache(bst_uint block_id) const {
  InMemoryBlock blk = read_block_from_disk(block_id);
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (!has_cached_block_locked(block_id)) {
    insert_block_into_cache_locked(std::move(blk));
  }
}

// evict_lru            
void ExternalMemoryDMatrix::evict_lru() const {
  // lru_order_ front = most recent, back = least recent
  if (lru_order_.empty()) return;
  bst_uint victim = lru_order_.back();
  lru_order_.pop_back();
  cache_.erase(victim);
}

bool ExternalMemoryDMatrix::has_cached_block_locked(bst_uint block_id) const {
  return cache_.find(block_id) != cache_.end();
}

void ExternalMemoryDMatrix::cleanup_completed_prefetch_locked() const {
  if (!prefetch_inflight_) return;
  if (prefetch_future_.valid() &&
      prefetch_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
    prefetch_future_.get();
    prefetch_inflight_ = false;
    prefetched_block_id_ = std::numeric_limits<bst_uint>::max();
  }
}

// get_block            
const InMemoryBlock& ExternalMemoryDMatrix::get_block(bst_uint block_id) const {
  std::future<void> wait_future;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cleanup_completed_prefetch_locked();
    if (prefetch_inflight_ && prefetched_block_id_ == block_id && prefetch_future_.valid()) {
      wait_future = std::move(prefetch_future_);
      prefetch_inflight_ = false;
      prefetched_block_id_ = std::numeric_limits<bst_uint>::max();
    }
  }
  if (wait_future.valid()) wait_future.wait();

  std::unique_lock<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(block_id);
  if (it != cache_.end()) {
    // Move to front of LRU (mark as recently used)
    auto pos = std::find(lru_order_.begin(), lru_order_.end(), block_id);
    if (pos != lru_order_.end()) {
    lru_order_.erase(pos);
    lru_order_.push_front(block_id);
    }
    return it->second;
  }

  // Cache miss — read without holding lock, then insert.
  // This keeps the cache mutex uncontended while disk IO is in progress.
  lock.unlock();
  InMemoryBlock blk = read_block_from_disk(block_id);

  lock.lock();
  auto after = cache_.find(block_id);
  if (after != cache_.end()) {
    auto pos = std::find(lru_order_.begin(), lru_order_.end(), block_id);
    if (pos != lru_order_.end()) {
      lru_order_.erase(pos);
      lru_order_.push_front(block_id);
    }
    return after->second;
  }
  insert_block_into_cache_locked(std::move(blk));
  return cache_.at(block_id);
}

// feature() — random access via block lookup       
bst_float ExternalMemoryDMatrix::feature(bst_uint global_row, bst_uint col) const {
  // Binary search over block_meta_ to find which block contains global_row
  bst_uint lo = 0, hi = static_cast<bst_uint>(block_meta_.size());
  while (lo + 1 < hi) {
    bst_uint mid = (lo + hi) / 2;
    if (block_meta_[mid].global_row_offset <= global_row) lo = mid;
    else hi = mid;
  }
  bst_uint block_id  = lo;
  bst_uint local_row = global_row - block_meta_[block_id].global_row_offset;
  const auto& blk = get_block(block_id);
  return blk.feature(local_row, col, n_cols_);
}

bst_float ExternalMemoryDMatrix::label(bst_uint global_row) const {
  bst_uint lo = 0, hi = static_cast<bst_uint>(block_meta_.size());
  while (lo + 1 < hi) {
    bst_uint mid = (lo + hi) / 2;
    if (block_meta_[mid].global_row_offset <= global_row) lo = mid;
    else hi = mid;
  }
  bst_uint local_row = global_row - block_meta_[lo].global_row_offset;
  return get_block(lo).labels[local_row];
}

// prefetch
void ExternalMemoryDMatrix::prefetch(bst_uint block_id) const {
  if (block_id >= num_blocks()) return;
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cleanup_completed_prefetch_locked();

  if (has_cached_block_locked(block_id)) return;
  if (prefetch_inflight_) {
    // Keep at most one in-flight prefetch to avoid IO storms.
    return;
  }

  prefetched_block_id_ = block_id;
  prefetch_inflight_ = true;
  prefetch_future_ = std::async(std::launch::async, [this, block_id]() {
    InMemoryBlock blk = read_block_from_disk(block_id);
    std::lock_guard<std::mutex> async_lock(cache_mutex_);
    if (!has_cached_block_locked(block_id)) {
      insert_block_into_cache_locked(std::move(blk));
    }
  });
}

void ExternalMemoryDMatrix::print_info() const {
  std::cout << "ExternalMemoryDMatrix: " << total_rows_
      << " rows × " << n_cols_ << " cols across "
      << block_meta_.size() << " disk blocks"
      << " (cache=" << cache_size_ << " blocks)\n";
}

//              
//  build_histograms_external  — streaming block-wise histogram build
//
//  Key insight: instead of accessing rows randomly (cache-unfriendly), we
//  iterate over blocks in order. For each block:
//  1. Build a boolean mask of which rows in this block are active.
//  2. Scan columns sequentially (column-major within the block).
//  3. Accumulate gradient statistics into histogram bins.
//
//  This preserves sequential IO from disk and maximises block reuse.
//              
std::vector<FeatureHistogram> build_histograms_external(
  const HistogramBuilder& builder,
  const ExternalMemoryDMatrix& dm,
  const std::vector<GradientPair>& grads,
  const std::vector<bst_uint>& row_indices,
  const std::vector<bst_uint>& col_indices)
{
  const bst_uint n_feat = dm.num_features();
  const auto& cut_points = builder.cut_points();

  // Initialise output histograms
  std::vector<FeatureHistogram> hists(n_feat);
  for (bst_uint f : col_indices) {
    if (f >= static_cast<bst_uint>(cut_points.size())) continue;
    bst_uint n_bins = static_cast<bst_uint>(cut_points[f].size()) + 1;
    hists[f].bins.assign(n_bins, HistogramBin{});
    hists[f].cut_points = cut_points[f];
  }

  // Build a global row mask: active_mask[r] = 1 if row r is in row_indices
  // O(n_active log n_active) build + O(1) per lookup
  std::vector<bst_uint> sorted_rows = row_indices;
  std::sort(sorted_rows.begin(), sorted_rows.end());

  // Iterate over blocks sequentially (stream from disk)
  const bst_uint n_blocks = dm.num_blocks();
  for (bst_uint b = 0; b < n_blocks; ++b) {
    const BlockMeta& meta = dm.block_meta(b);
    const InMemoryBlock& blk = dm.get_block(b);

    // Prefetch next block while we process this one
    if (b + 1 < n_blocks) dm.prefetch(b + 1);

    const bst_uint row_offset = meta.global_row_offset;

    // For each row in this block, check if it is active
    for (bst_uint local_r = 0; local_r < meta.n_rows; ++local_r) {
    bst_uint global_r = row_offset + local_r;

    // Check membership via binary search in sorted_rows
    bool active = std::binary_search(
      sorted_rows.begin(), sorted_rows.end(), global_r);
    if (!active) continue;

    const GradientPair& gp = grads[global_r];

    // Accumulate into each requested feature's histogram
    for (bst_uint f : col_indices) {
      bst_float val = blk.feature(local_r, f, meta.n_cols);
      const auto& cuts = hists[f].cut_points;

      // Binary search for bin boundary
      bst_uint bin = static_cast<bst_uint>(
        std::upper_bound(cuts.begin(), cuts.end(), val) - cuts.begin());
      bin = std::min(bin, static_cast<bst_uint>(hists[f].bins.size() - 1));
      hists[f].bins[bin].add(gp);
    }
    }
  }

  return hists;
}

} // namespace xgb
