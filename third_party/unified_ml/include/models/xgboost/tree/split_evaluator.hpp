#pragma once
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include <vector>

namespace xgb {

class DMatrix;

//                        
//  SplitEvaluator
//  Finds the best split for a given set of instances
//  and gradient statistics.
//
//  Implements:
//    Algorithm 1  — Exact Greedy
//    Algorithm 2  — Approximate (histogram-based)
//    Algorithm 3  — Sparsity-aware
//    Feature 5    — L1 regularisation (alpha)
//                        
class SplitEvaluator {
public:
    explicit SplitEvaluator(const TreeConfig& cfg) : cfg_(cfg) {}

    //   Exact greedy split (Algorithm 1)    
    // Also dispatches to find_best_split_approx when cfg_.use_approx is set.
    SplitCandidate find_best_split(
        const DMatrix& dm,
        const std::vector<GradientPair>& grads,
        const std::vector<bst_uint>& row_indices,
        const std::vector<bst_uint>& col_indices) const;

    //   Approximate split (Algorithm 2)    
    // Builds a fresh HistogramBuilder internally, computes cut points via
    // weighted quantile sketch, then evaluates splits on histograms.
    SplitCandidate find_best_split_approx(
        const DMatrix& dm,
        const std::vector<GradientPair>& grads,
        const std::vector<bst_uint>& row_indices,
        const std::vector<bst_uint>& col_indices) const;

    //   Approximate split from pre-built histograms             
    // Used when DecisionTree passes down pre-computed parent histograms
    // (histogram subtraction optimisation, Feature 3).
    SplitCandidate find_best_split_approx_from_hists(
        const std::vector<FeatureHistogram>& hists,
        double G_total,
        double H_total,
        const std::vector<bst_uint>& col_indices) const;

    //   Sparsity-aware split (Algorithm 3)   
    SplitCandidate find_best_split_sparse(
        const DMatrix& dm,
        const std::vector<GradientPair>& grads,
        const std::vector<bst_uint>& row_indices,
        bst_uint feature_idx,
        double G_total, double H_total) const;

private:
    TreeConfig cfg_;

    //   Feature 5: L1-aware scoring                    
    // node_score(G, H) = max(0, |G| - α)² / (H + λ)
    // When α = 0: reduces to G² / (H + λ).
    bst_float node_score(double G, double H) const;

    // split_gain = ½·[score(L) + score(R) - score(parent)] - γ
    bst_float split_gain(double GL, double HL, double GR, double HR) const;

    // Alias for compatibility
    bst_float score_split(double GL, double HL, double GR, double HR) const;
};

} // namespace xgb
