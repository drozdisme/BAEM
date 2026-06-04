#pragma once
#include "models/xgboost/tree/tree_node.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <memory>
#include <string>

namespace xgb {

class DMatrix;

//                        
//  DecisionTree
//  A single regression tree (one boosting iteration).
//  Nodes are stored in a flat array indexed by NodeId.
//  Root is always at index 0.
//
//  Growth policy: level-wise (best-first also planned).
//  Corresponds to the ensemble component fk in Eq. (1).
//                        
class DecisionTree {
public:
    explicit DecisionTree(const TreeConfig& cfg);

    // Build the tree from gradient pairs    
    void build(const DMatrix& dm,
               const std::vector<GradientPair>& grads,
               const std::vector<bst_uint>& row_indices,
               const std::vector<bst_uint>& col_indices);

    // Deserialisation injection         
    // Moves a pre-built node array into this tree (used by load_model).
    void load_nodes(std::vector<TreeNode> nodes) {
        nodes_ = std::move(nodes);
    }

    // Scale all leaf values by a factor (used to bake eta at train time)  
    void scale_leaves_by(bst_float factor);

    // Predict one sample            
    bst_float predict(const DMatrix& dm, bst_uint row) const;

    // Batch predict               
    std::vector<bst_float> predict_batch(
        const DMatrix& dm,
        const std::vector<bst_uint>& row_indices) const;

    // Export                  
    std::string dump_text() const;
    std::string dump_json() const;

    void compute_feature_importance(
        std::vector<bst_float>& importance,
        bst_uint n_features) const;

    // Accessors                 
    const std::vector<TreeNode>& nodes()       const { return nodes_; }
    bst_int  num_leaves()  const;
    bst_int  actual_depth() const;

private:
    TreeConfig cfg_;
    std::vector<TreeNode> nodes_;

    // HistogramBuilder reused across levels for approximate split finding
    std::unique_ptr<HistogramBuilder> hist_builder_;

    // Internal growth              
    NodeId alloc_node(NodeId parent, bst_int depth);
    void   make_leaf(NodeId nid, double sum_grad, double sum_hess);

    // Recursive build (parent_hists=nullptr → exact path or root of approx)
    void build_node(
        NodeId nid,
        const DMatrix& dm,
        const std::vector<GradientPair>& grads,
        std::vector<bst_uint>& row_indices,
        const std::vector<bst_uint>& col_indices,
        std::vector<FeatureHistogram>* parent_hists = nullptr);

    // Feature 5: elastic-net leaf weight  w* = -sign(G)·max(0,|G|-α)/(H+λ)
    static bst_float calc_leaf_weight(double sum_grad, double sum_hess,
                                       bst_float lambda, bst_float alpha = 0.f);

    static bst_float calc_gain(double sum_grad, double sum_hess,
                                bst_float lambda);

    static bst_float calc_split_gain(double GL, double HL,
                                      double GR, double HR,
                                      bst_float lambda, bst_float gamma);
};

} // namespace xgb
