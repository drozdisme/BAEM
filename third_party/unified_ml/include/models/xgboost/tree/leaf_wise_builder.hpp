#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/tree/leaf_wise_builder.h
//
//  Feature 9: Best-First (Leaf-wise) Tree Growth Strategy
//
//  This builder implements the lossguide / best-first growth policy used
//  by LightGBM and optionally available in XGBoost.
//
//  Algorithm:
//    1. Initialise a max-heap of SplitCandidate objects for the root node.
//    2. At each step pop the candidate with the highest gain.
//    3. Split that leaf, compute candidates for both children.
//    4. Push children onto the heap.
//    5. Stop when the heap is empty, max_leaves is reached, or gain ≤ gamma.
//
//  Unlike level-wise (depth-wise) growth, leaf-wise focuses budget on the
//  leaves that offer the greatest reduction in loss, which often produces
//  better trees with fewer nodes for the same budget.
//
//  Key config parameter:  max_leaves  (e.g. 31 → similar to default LightGBM)
//  Compatible with: max_depth as an additional hard constraint.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/tree/tree_node.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include <vector>
#include <queue>
#include <memory>

namespace xgb {

class DMatrix;

//                                       
//  LeafSplitCandidate
//  A split candidate for a specific leaf in the heap.
//                                       
struct LeafSplitCandidate {
    NodeId      node_id     {kInvalidNodeId};
    SplitCandidate split;                        // best split for this leaf
    std::vector<bst_uint> row_indices;           // data rows in this leaf

    // Max-heap comparison: higher gain = higher priority
    bool operator<(const LeafSplitCandidate& rhs) const {
        return split.gain < rhs.split.gain;   // inverted for max-heap
    }
};

//                                       
//  LeafWiseBuilder
//
//  Builds a single tree using the best-first (lossguide) growth policy.
//  Integrates with the existing DecisionTree node array layout.
//                                       
class LeafWiseBuilder {
public:
    // tree_nodes  : node array owned by the caller (DecisionTree)
    // cfg         : tree hyperparameters
    LeafWiseBuilder(std::vector<TreeNode>& tree_nodes,
                    const TreeConfig&      cfg);

    // Build the tree from gradient pairs.
    // On return, tree_nodes is fully populated.
    void build(const DMatrix&                   dm,
               const std::vector<GradientPair>& grads,
               const std::vector<bst_uint>&     row_indices,
               const std::vector<bst_uint>&     col_indices);

private:
    std::vector<TreeNode>&  nodes_;
    const TreeConfig&       cfg_;
    std::unique_ptr<HistogramBuilder> hist_builder_;

    // max_leaves (derived from cfg_ or set directly)
    bst_int max_leaves_;

    //   Node allocation                           
    NodeId alloc_node(NodeId parent, bst_int depth);
    void   make_leaf(NodeId nid, double sum_grad, double sum_hess);

    //   Split finding                            
    LeafSplitCandidate find_best_split(
        NodeId                           nid,
        const DMatrix&                   dm,
        const std::vector<GradientPair>& grads,
        const std::vector<bst_uint>&     row_indices,
        const std::vector<bst_uint>&     col_indices) const;

    //   Apply a split                            
    std::pair<std::vector<bst_uint>, std::vector<bst_uint>>
    partition_rows(const DMatrix&             dm,
                   const std::vector<bst_uint>& rows,
                   bst_uint                   feature,
                   bst_float                  split_value,
                   bool                       default_left) const;

    static bst_float calc_gain(double sum_grad, double sum_hess, bst_float lambda);
    static bst_float calc_leaf_weight(double sum_grad, double sum_hess,
                                      bst_float lambda, bst_float alpha = 0.f);
    static bst_float calc_split_gain(double GL, double HL, double GR, double HR,
                                     bst_float lambda, bst_float gamma);
};

} // namespace xgb
