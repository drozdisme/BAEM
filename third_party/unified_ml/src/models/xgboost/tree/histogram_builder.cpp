#include "models/xgboost/tree/histogram_builder.hpp"
#include "models/xgboost/tree/weighted_quantile_sketch.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include <algorithm>
#include <numeric>
#include <cassert>

namespace xgb {

// FeatureHistogram          

void FeatureHistogram::clear() {
  for (auto& b : bins) { b.sum_grad = 0; b.sum_hess = 0; b.count = 0; }
}

HistogramBin FeatureHistogram::prefix_sum(bst_uint bin_idx) const {
  HistogramBin acc;
  for (bst_uint i = 0; i <= bin_idx && i < static_cast<bst_uint>(bins.size()); ++i) {
    acc.sum_grad += bins[i].sum_grad;
    acc.sum_hess += bins[i].sum_hess;
    acc.count  += bins[i].count;
  }
  return acc;
}

// HistogramBuilder          

//              
//  build_cut_points  — Feature 1: True Weighted Quantile Sketch
//
//  Algorithm (XGBoost paper §3.3, Appendix):
//
//  1. For each feature f, collect all (feature_value, hessian) pairs.
//   The hessian h_i is the sample weight: instances with higher curvature
//   should influence split boundaries more strongly.
//
//  2. Feed pairs into WeightedQuantileSketch(eps = cfg_.sketch_eps).
//   The sketch maintains a GK-style summary and prunes entries to stay
//   within the ε-error bound.
//
//  3. Extract n_bins quantile cut points. Each bin covers ≤ ε fraction
//   of total Hessian mass W = Σ h_i, guaranteeing:
//   rank_error ≤ ε·W ≤ ε·n (squared loss: h_i = 2 ≥ 1)
//
//  Acceptance criterion: for ε = 0.05, rank_error ≤ 0.05·n. ✓
//              
void HistogramBuilder::build_cut_points(
  const DMatrix& dm,
  const std::vector<GradientPair>& grads,
  bst_uint n_bins)
{
  const bst_uint n_feat = dm.num_features();
  const bst_uint n_rows = dm.num_rows();
  cut_points_.resize(n_feat);

  // Use at most n_rows bins (can't have more quantiles than samples)
  const bst_uint effective_bins = std::min(n_bins, n_rows);

  for (bst_uint f = 0; f < n_feat; ++f) {
    // Step 1: build per-feature weighted quantile sketch   
    WeightedQuantileSketch sketch(cfg_.sketch_eps);

    for (bst_uint r = 0; r < n_rows; ++r) {
    bst_float val = dm.feature(r, f);
    // Hessian as weight; clip to non-negative (convex loss guarantee)
    bst_float w = std::max(0.f, grads[r].hess);
    sketch.push(val, w);
    }

    // Step 2: finalize — builds GK summary and prunes    
    sketch.finalize();

    // Step 3: extract cut points from sketch     
    std::vector<bst_float> cuts = sketch.get_cut_points(effective_bins);

    // Remove exact duplicates (arise from flat / constant features)
    auto last = std::unique(cuts.begin(), cuts.end());
    cuts.erase(last, cuts.end());

    cut_points_[f] = std::move(cuts);
  }
}

// Internal: binary search for bin index       
bst_uint HistogramBuilder::feature_to_bin(
  bst_uint feature_idx, bst_float value) const
{
  const auto& cuts = cut_points_[feature_idx];
  if (cuts.empty()) return 0;
  // upper_bound returns iterator to first cut strictly > value
  auto it = std::upper_bound(cuts.begin(), cuts.end(), value);
  bst_uint bin = static_cast<bst_uint>(std::distance(cuts.begin(), it));
  // bins are [0 .. cuts.size()]; clamp to last bin
  return std::min(bin, static_cast<bst_uint>(cuts.size()));
}

// build_histograms          
std::vector<FeatureHistogram> HistogramBuilder::build_histograms(
  const DMatrix& dm,
  const std::vector<GradientPair>& grads,
  const std::vector<bst_uint>& row_indices,
  const std::vector<bst_uint>& col_indices) const
{
  std::vector<FeatureHistogram> hists(dm.num_features());

  // Initialise bin arrays for each requested column
  for (bst_uint f : col_indices) {
    bst_uint n_bins = static_cast<bst_uint>(cut_points_[f].size()) + 1;
    hists[f].bins.assign(n_bins, HistogramBin{});
    hists[f].cut_points = cut_points_[f];
  }

  // Scatter gradient/Hessian statistics into bins
  for (bst_uint row : row_indices) {
    const GradientPair& gp = grads[row];
    for (bst_uint f : col_indices) {
    bst_float val = dm.feature(row, f);
    bst_uint  bin = feature_to_bin(f, val);
    if (bin < static_cast<bst_uint>(hists[f].bins.size()))
      hists[f].bins[bin].add(gp);
    }
  }
  return hists;
}

// subtract_histograms           
//  Compute child_small = parent - child_large element-wise.
//  Requires both histograms to share the same bin structure (same cut points).
void HistogramBuilder::subtract_histograms(
  const FeatureHistogram& parent,
  const FeatureHistogram& sibling,
  FeatureHistogram& child) const
{
  assert(parent.bins.size() == sibling.bins.size()
     && "parent/sibling must have identical bin count");

  child.cut_points = parent.cut_points;
  child.bins.resize(parent.bins.size());

  for (bst_uint i = 0; i < static_cast<bst_uint>(parent.bins.size()); ++i) {
    child.bins[i].sum_grad = parent.bins[i].sum_grad - sibling.bins[i].sum_grad;
    child.bins[i].sum_hess = parent.bins[i].sum_hess - sibling.bins[i].sum_hess;
    child.bins[i].count  =
    (parent.bins[i].count >= sibling.bins[i].count)
      ? (parent.bins[i].count - sibling.bins[i].count)
      : 0u;
  }
}

} // namespace xgb

//              
//  Feature 3: build_histograms_blocked  — cache-aware block iteration
//
//  Standard build_histograms() scans rows in arbitrary order (as given by
//  row_indices), causing random accesses into the gradient array and the
//  feature matrix.  For large n this thrashes L1/L2 cache.
//
//  With ColumnBlockStore, each feature's data is split into fixed-size sorted
//  chunks (ColumnBlock).  We iterate blocks of one feature at a time:
//
//  for each feature f in col_indices:
//  for each ColumnBlock of feature f:
//    for each SortedEntry (value, row) in the block:
//    if row is in the active set → accumulate grads[row] into bin
//
//  Benefits:
//  • The inner loop accesses blk.entries[] sequentially → cache prefetcher
//  streams the (value, row) pairs without stalls.
//  • The grads[] access is still random, but amortised: each cache line of
//  grads[] (16 floats × 8 B = 128 B) is reused across many entries in the
//  same block when block_size >> cache_line_span.
//  • No sort needed inside build_histograms — blocks are pre-sorted.
//
//  Active-row check:
//  We build a boolean presence vector (row_set) of size n_rows once, then
//  check row_set[entry.row_idx] in O(1) per entry.
//  Building row_set is O(|row_indices|).
//              
#include "models/xgboost/data/column_block.hpp"

namespace xgb {

std::vector<FeatureHistogram> HistogramBuilder::build_histograms_blocked(
  const ColumnBlockStore& store,
  const std::vector<GradientPair>& grads,
  const std::vector<bst_uint>& row_indices,
  const std::vector<bst_uint>& col_indices) const
{
  // Build O(n) boolean presence set      
  // row_set[i] = true iff row i is in row_indices for this node.
  // We avoid std::unordered_set to keep this allocation cheap and linear.
  const bst_uint n_rows = static_cast<bst_uint>(grads.size());
  std::vector<bool> row_set(n_rows, false);
  for (bst_uint r : row_indices) {
    if (r < n_rows) row_set[r] = true;
  }

  // Initialise output histograms        
  const bst_uint n_feat = store.num_features();
  std::vector<FeatureHistogram> hists(n_feat);

  for (bst_uint f : col_indices) {
    if (f >= static_cast<bst_uint>(cut_points_.size())) continue;
    bst_uint n_bins = static_cast<bst_uint>(cut_points_[f].size()) + 1;
    hists[f].bins.assign(n_bins, HistogramBin{});
    hists[f].cut_points = cut_points_[f];
  }

  // Iterate blocks per feature        
  for (bst_uint f : col_indices) {
    if (f >= store.num_features()) continue;
    if (f >= static_cast<bst_uint>(cut_points_.size())) continue;

    auto& hist = hists[f];
    const auto& cuts = hist.cut_points;

    // Iterate each sorted block for feature f
    for (const ColumnBlock& blk : store.blocks_for(f)) {
    // Sequential scan — CPU prefetcher can stream ahead automatically
    const SortedEntry* entries = blk.data();
    const bst_uint   n_ents  = blk.size();

    for (bst_uint e = 0; e < n_ents; ++e) {
      const bst_uint  row = entries[e].row_idx;
      const bst_float val = entries[e].value;

      // Skip rows not in the active set
      if (row >= n_rows || !row_set[row]) continue;

      // Binary search for histogram bin
      // upper_bound is cache-friendly here: cuts[] fits in a few
      // cache lines for typical 256-bin histograms.
      bst_uint bin = static_cast<bst_uint>(
        std::upper_bound(cuts.begin(), cuts.end(), val) - cuts.begin());
      bin = std::min(bin, static_cast<bst_uint>(hist.bins.size() - 1));

      hist.bins[bin].add(grads[row]);
    }
    }
  }

  return hists;
}

} // namespace xgb
