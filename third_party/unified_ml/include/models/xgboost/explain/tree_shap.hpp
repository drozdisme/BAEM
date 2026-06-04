#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/explain/tree_shap.h
//
//  TreeSHAP — exact Shapley value computation for tree ensembles.
//
//  Reference: Lundberg et al., "Consistent Individualized Feature
//             Attribution for Tree Ensembles" (2019).
//
//  Complexity: O(T · L · d)
//    T = number of trees
//    L = number of leaves per tree
//    d = maximum tree depth
//
//  The algorithm propagates "path weights" through the tree, tracking
//  how each feature on the path from root to a leaf contributes to the
//  final leaf value.  The contribution is the Shapley value of that
//  feature for that sample.
//
//  Usage:
//    TreeExplainer explainer(booster);
//    auto shap = explainer.shap_values(dm, row_index);
//    // shap[j] = Shapley value of feature j for that prediction
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/tree/decision_tree.hpp"
#include <vector>
#include <memory>
#include <string>

namespace xgb {

class GradientBooster;
class DMatrix;

//                                       
//  PathElement
//  Internal: one entry on the root-to-leaf decision path.
//  Tracks whether the feature at this node was "present" in the active set S.
//                                       
struct PathElement {
    bst_int   feature_index  {-1};   // -1 = leaf / bias node
    bst_float zero_fraction  {0.f};  // fraction of training data going left (≈ cover_left/cover)
    bst_float one_fraction   {0.f};  // fraction going to the active child
    bst_float weight         {0.f};  // path weight
};

//                                       
//  TreeShap
//  Computes Shapley values for a SINGLE tree.
//                                       
class TreeShap {
public:
    explicit TreeShap(const DecisionTree& tree, bst_uint num_features);

    // Compute SHAP values for one sample.
    // feature_values: feature vector for this sample (length = num_features_)
    // out: output vector of length num_features_ + 1
    //      out[0..num_features_-1] = per-feature Shapley values
    //      out[num_features_]      = bias (expected prediction = phi_0)
    void compute(const std::vector<bst_float>& feature_values,
                 std::vector<double>&           out) const;

private:
    const DecisionTree& tree_;
    bst_uint            num_features_;

    //   Core recursive algorithm                      

    // Extend the active path by one element.
    static void extend_path(std::vector<PathElement>& path,
                            bst_float zero_fraction,
                            bst_float one_fraction,
                            bst_int   feature_index);

    // Remove element from the path when backtracking.
    static void unwind_path(std::vector<PathElement>& path, bst_uint path_index);

    // Compute the unnormalised sum of path weights for a particular path_index.
    static bst_float unwound_path_sum(const std::vector<PathElement>& path,
                                      bst_uint path_index);

    // Recursive tree walk.
    void recurse(NodeId                        node_id,
                 const std::vector<bst_float>& fval,
                 std::vector<PathElement>      path,
                 bst_float                     path_weight,
                 std::vector<double>&          out) const;

    // Get coverage fraction for a node (approximated from TreeNodeStats).
    bst_float node_coverage(NodeId nid) const;
};

//                                       
//  TreeExplainer
//  Ensemble-level explainer: sums SHAP contributions across all trees.
//                                       
class TreeExplainer {
public:
    // booster must outlive this object.
    explicit TreeExplainer(const GradientBooster& booster, bst_uint num_features);

    // Compute mean |SHAP| per feature across all rows (global importance).
    std::vector<double> mean_abs_shap(const DMatrix& dm) const;

    // Shapley values for a single sample.
    // Returns vector of length num_features_ + 1.
    // The last element is the bias (phi_0 = expected model output).
    std::vector<double> shap_values(const DMatrix& dm, bst_uint row) const;

    // Batch: returns [n_rows x (num_features+1)] flattened row-major.
    std::vector<double> shap_values_batch(const DMatrix& dm) const;

    bst_uint num_features() const { return num_features_; }

private:
    const GradientBooster& booster_;
    bst_uint               num_features_;

    // One TreeShap object per tree in the ensemble.
    std::vector<TreeShap> tree_shapers_;

    // Build tree_shapers_ from booster trees.
    void init_shapers();
};

} // namespace xgb
