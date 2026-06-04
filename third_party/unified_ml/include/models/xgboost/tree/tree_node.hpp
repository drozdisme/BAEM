#pragma once
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <string>

namespace xgb {

//                        
//  TreeNode
//  Represents one node in a CART regression tree.
//  Both internal (split) nodes and leaf nodes.
//
//  Internal node layout (Sec. 2.2):
//    - feature_idx, split_value, default_left
//    - child_left, child_right
//  Leaf node:
//    - leaf_value = w*  (Eq. 5)
//    - is_leaf = true
//                        
struct TreeNode {
    // Identity                 
    NodeId id          {kInvalidNodeId};
    bst_int depth      {0};

    // Split info (valid when !is_leaf)     
    bst_uint   feature_idx   {0};
    bst_float  split_value   {0.f};
    bool       default_left  {true};   // missing-value direction (Sec. 3.4)

    // Children (valid when !is_leaf)      
    NodeId child_left   {kInvalidNodeId};
    NodeId child_right  {kInvalidNodeId};
    NodeId parent       {kInvalidNodeId};

    // Leaf value (valid when is_leaf)      
    bst_float  leaf_value {0.f};

    // Node type                 
    bool is_leaf  {false};
    bool is_valid {true};   // false for pruned / deleted nodes

    // Gradient statistics (kept for analysis)  
    double sum_grad {0.0};
    double sum_hess {0.0};
    bst_uint n_samples {0};

    // Split gain achieved at this node     
    bst_float split_gain {0.f};

    // Helpers                  
    bool is_root() const { return parent == kInvalidNodeId; }

    bool goes_left(bst_float feature_val, bool is_missing) const {
        if (is_missing)   return default_left;
        return feature_val <= split_value;
    }

    std::string to_string() const;
};

} // namespace xgb
