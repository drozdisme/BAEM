#include "models/xgboost/tree/tree_node.hpp"
#include <sstream>

namespace xgb {

std::string TreeNode::to_string() const {
  std::ostringstream oss;
  if (is_leaf) {
    oss << "Leaf[id=" << id << " val=" << leaf_value
    << " n=" << n_samples << "]";
  } else {
    oss << "Node[id=" << id << " f" << feature_idx
    << "<=" << split_value << " gain=" << split_gain << "]";
  }
  return oss.str();
}

} // namespace xgb
