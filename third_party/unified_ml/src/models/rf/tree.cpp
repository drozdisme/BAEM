#include "models/rf/tree.hpp"
#include <numeric>
#include <stdexcept>

namespace rf {

void FeatureImportance::accumulate(const TreeNode* node) {
  if (!node) return;
  if (node->is_internal()) {
    size_t fi = static_cast<size_t>(node->feature_index);
    if (fi < n_features) {
    scores[fi] += node->impurity_decrease * static_cast<double>(node->n_samples);
    }
    accumulate(node->left.get());
    accumulate(node->right.get());
  }
}

void FeatureImportance::normalize() {
  double total = 0.0;
  for (double s : scores) total += s;
  if (total > 0.0) {
    for (double& s : scores) s /= total;
  }
}

} // namespace rf
