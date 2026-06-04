#pragma once
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/core/config.hpp"
// Forward declaration — avoids circular include
namespace xgb { class ColumnBlockStore; }
#include <vector>
#include <memory>

namespace xgb {

class DMatrix;

//                        
//  HistogramBin  — one bucket in a feature histogram
//  Stores aggregated (G, H) for all samples falling
//  into this bin.  Used in approximate split finding.
//                        
struct HistogramBin {
    double sum_grad{0.0};
    double sum_hess{0.0};
    bst_uint count{0};

    void add(const GradientPair& gp) {
        sum_grad += gp.grad;
        sum_hess += gp.hess;
        ++count;
    }
};

//                        
//  FeatureHistogram  — histogram for one feature
//  bins_.size() == n_bins
//  cut_points_[i] = upper boundary of bin i
//                        
struct FeatureHistogram {
    std::vector<HistogramBin> bins;
    std::vector<bst_float>    cut_points;   // quantile boundaries

    bst_uint n_bins() const { return static_cast<bst_uint>(bins.size()); }

    // Reset all bins (zero-out)
    void clear();

    // Prefix-sum for split finding (left accumulation)
    HistogramBin prefix_sum(bst_uint bin_idx) const;
};

//                        
//  HistogramBuilder
//  Builds gradient histograms used in approximate
//  split finding (Algorithm 2, Sec. 3.2).
//
//  Also forms the foundation for the "hist" tree_method
//  which is the fastest variant in real XGBoost.
//
//  Future: GPU histogram, cache-aware block structure
//          (Sec. 4.1), parent-child subtraction trick.
//                        
class HistogramBuilder {
public:
    explicit HistogramBuilder(const TreeConfig& cfg) : cfg_(cfg) {}

    // Build cut points via weighted quantile sketch (Sec. 3.3)
    void build_cut_points(const DMatrix& dm,
                          const std::vector<GradientPair>& grads,
                          bst_uint n_bins = 256);

    // Build histograms for all features for a given node
    std::vector<FeatureHistogram> build_histograms(
        const DMatrix& dm,
        const std::vector<GradientPair>& grads,
        const std::vector<bst_uint>& row_indices,
        const std::vector<bst_uint>& col_indices) const;

    // Subtraction trick: child = parent - sibling
    // Avoids rebuilding one child's histogram from scratch.
    void subtract_histograms(const FeatureHistogram& parent,
                             const FeatureHistogram& sibling,
                             FeatureHistogram& child) const;

    const std::vector<std::vector<bst_float>>& cut_points() const {
        return cut_points_;
    }

    //   Feature 3: Cache-aware column block histogram build         
    // Iterates ColumnBlockStore blocks sequentially instead of random row access.
    // Each block is sorted by feature value → natural left-to-right accumulation.
    // Complexity: O(nnz_per_node) with sequential cache lines.
    std::vector<FeatureHistogram> build_histograms_blocked(
        const class ColumnBlockStore& store,
        const std::vector<GradientPair>& grads,
        const std::vector<bst_uint>& row_indices,
        const std::vector<bst_uint>& col_indices) const;

private:
    TreeConfig cfg_;
    std::vector<std::vector<bst_float>> cut_points_;   // [feature][bin]

    bst_uint feature_to_bin(bst_uint feature_idx, bst_float value) const;
};

} // namespace xgb
