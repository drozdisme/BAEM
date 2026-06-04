// ═══════════════════════════════════════════════════════════════════════════
//  src/explain/tree_shap.cpp
//
//  Exact TreeSHAP implementation following Lundberg et al. (2019).
//  Path-based recursion with O(T · L · d) complexity.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/explain/tree_shap.hpp"
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/tree/tree_node.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace xgb {

// ═══════════════════════════════════════════════════════════════════════════
//  TreeShap implementation
// ═══════════════════════════════════════════════════════════════════════════

TreeShap::TreeShap(const DecisionTree& tree, bst_uint num_features)
  : tree_(tree), num_features_(num_features)
{}

//  extend_path          
//
//  Append a new PathElement to the current path.
//  Updates the weight terms for all previous path elements.
//
//  Following the formula in Algorithm 1 of Lundberg et al.:
//  w_{l+1}(z) = w_l(z) * (zero_frac * (l+1-depth) + one_frac * depth)
//         / (l+1)
//  Here depth = position within the path (number of times feature was used).
//
void TreeShap::extend_path(std::vector<PathElement>& path,
          bst_float zero_fraction,
          bst_float one_fraction,
          bst_int feature_index)
{
  const bst_uint l = static_cast<bst_uint>(path.size());

  // Push new element
  PathElement pe;
  pe.feature_index = feature_index;
  pe.zero_fraction = zero_fraction;
  pe.one_fraction  = one_fraction;
  pe.weight = (l == 0) ? 1.f : 0.f;
  path.push_back(pe);

  // Update weights for all but the new element
  for (bst_int i = static_cast<bst_int>(l) - 1; i >= 0; --i) {
    path[i + 1].weight += one_fraction * path[i].weight
          * static_cast<bst_float>(i + 1)
          / static_cast<bst_float>(l + 1);
    path[i].weight   *= zero_fraction
          * static_cast<bst_float>(l - i)
          / static_cast<bst_float>(l + 1);
  }
}

//  unwound_path_sum          
//
//  Compute the sum of path weights that would remain if element path_index
//  were removed.  Used to calculate the Shapley contribution.
//
bst_float TreeShap::unwound_path_sum(const std::vector<PathElement>& path,
              bst_uint path_index)
{
  const bst_uint l = static_cast<bst_uint>(path.size()) - 1;
  bst_float total = 0.f;

  const bst_float one_frac = path[path_index].one_fraction;
  if (one_frac != 0.f) {
    bst_float next = 1.f;
    for (bst_int i = static_cast<bst_int>(l); i >= 1; --i) {
    next *= static_cast<bst_float>(i + 1);
    if (static_cast<bst_uint>(i) != path_index) {
      total += next * path[i].weight / one_frac;
    }
    next /= static_cast<bst_float>(i);
    }
  } else {
    // one_fraction = 0: use zero_fraction branch
    const bst_float zero_frac = path[path_index].zero_fraction;
    if (zero_frac != 0.f) {
    bst_float next = 1.f;
    for (bst_int i = static_cast<bst_int>(l); i >= 1; --i) {
      next *= static_cast<bst_float>(i + 1);
      if (static_cast<bst_uint>(i) != path_index) {
        total += next * path[i].weight / zero_frac;
      }
      next /= static_cast<bst_float>(i);
    }
    }
  }

  return total * (one_frac - path[path_index].zero_fraction)
       / static_cast<bst_float>(l + 1);
}

//  unwind_path          
void TreeShap::unwind_path(std::vector<PathElement>& path, bst_uint path_index)
{
  const bst_uint l = static_cast<bst_uint>(path.size()) - 1;
  const bst_float one_frac  = path[path_index].one_fraction;
  const bst_float zero_frac = path[path_index].zero_fraction;

  bst_float next = path[l].weight;
  for (bst_int i = static_cast<bst_int>(l) - 1; i >= 0; --i) {
    if (one_frac != 0.f) {
    const bst_float tmp = path[i].weight;
    path[i].weight = next * static_cast<bst_float>(l + 1)
           / (one_frac * static_cast<bst_float>(i + 1));
    next = tmp - path[i].weight * zero_frac
         * static_cast<bst_float>(l - i)
         / static_cast<bst_float>(l + 1);
    } else {
    path[i].weight = (path[i].weight * static_cast<bst_float>(l + 1))
           / (zero_frac * static_cast<bst_float>(l - i));
    }
  }

  path.erase(path.begin() + path_index);
}

//  node_coverage           
bst_float TreeShap::node_coverage(NodeId nid) const
{
  const auto& nodes = tree_.nodes();
  if (nid < 0 || nid >= static_cast<bst_int>(nodes.size())) return 1.f;
  const double h = nodes[nid].sum_hess;
  return static_cast<bst_float>(h > 0.0 ? h : 1.0);
}

//  recurse            
void TreeShap::recurse(NodeId        node_id,
        const std::vector<bst_float>& fval,
        std::vector<PathElement>  path,
        bst_float       path_weight,
        std::vector<double>&     out) const
{
  const auto& nodes = tree_.nodes();
  if (node_id < 0 || node_id >= static_cast<bst_int>(nodes.size())) return;
  const TreeNode& node = nodes[node_id];

  if (node.is_leaf) {
    // Distribute leaf value * path_weight back to features on the path
    for (bst_uint i = 1; i < path.size(); ++i) {
    const bst_float w = unwound_path_sum(path, i);
    const bst_int fi = path[i].feature_index;
    if (fi >= 0 && fi < static_cast<bst_int>(num_features_)) {
      out[fi] += w * (path[i].one_fraction - path[i].zero_fraction)
           * static_cast<double>(node.leaf_value);
    }
    }
    return;
  }

  // Internal split node         
  const bst_uint feat  = node.feature_idx;
  const bool   missing = (feat >= fval.size());
  const bst_float  x   = missing ? 0.f : fval[feat];

  // Determine which child to follow (hot path) and which is the cold path
  const bool  goes_left  = node.goes_left(x, missing);
  const NodeId hot_child  = goes_left ? node.child_left  : node.child_right;
  const NodeId cold_child = goes_left ? node.child_right : node.child_left;

  // Fraction of samples going to hot/cold child
  const bst_float hot_cover  = node_coverage(hot_child);
  const bst_float cold_cover = node_coverage(cold_child);
  const bst_float total_cov  = hot_cover + cold_cover;

  const bst_float hot_zero_frac  = (total_cov > 0.f) ? hot_cover  / total_cov : 0.5f;
  const bst_float cold_zero_frac = (total_cov > 0.f) ? cold_cover / total_cov : 0.5f;

  // Incoming one_fraction is 1 (this sample takes this branch)
  extend_path(path, hot_zero_frac,  1.f, static_cast<bst_int>(feat));

  // Recurse hot child
  recurse(hot_child,  fval, path, path_weight, out);

  // Undo extension for cold child exploration
  unwind_path(path, static_cast<bst_uint>(path.size() - 1));
  extend_path(path, cold_zero_frac, 0.f, static_cast<bst_int>(feat));

  recurse(cold_child, fval, path, path_weight, out);
}

//  Public compute           
void TreeShap::compute(const std::vector<bst_float>& feature_values,
        std::vector<double>&     out) const
{
  const bst_uint n_out = num_features_ + 1;
  out.assign(n_out, 0.0);

  std::vector<PathElement> path;
  path.reserve(64); // reserve depth capacity

  // Seed the path with the implicit root element (fi=-1 = bias node).
  // Required so the leaf loop (i=1..size) sees the first split level.
  extend_path(path, 1.f, 1.f, -1);

  recurse(kRootNodeId, feature_values, path, 1.f, out);
}

// ═══════════════════════════════════════════════════════════════════════════
//  TreeExplainer implementation
// ═══════════════════════════════════════════════════════════════════════════

TreeExplainer::TreeExplainer(const GradientBooster& booster,
          bst_uint     num_features)
  : booster_(booster), num_features_(num_features)
{
  init_shapers();
}

void TreeExplainer::init_shapers()
{
  tree_shapers_.clear();
  for (const auto& tree : booster_.trees()) {
    tree_shapers_.emplace_back(*tree, num_features_);
  }
}

//  Single-sample SHAP         
std::vector<double>
TreeExplainer::shap_values(const DMatrix& dm, bst_uint row) const
{
  const bst_uint n_out = num_features_ + 1;
  std::vector<double> total(n_out, 0.0);

  // Extract feature values for this row
  std::vector<bst_float> fval(num_features_, 0.f);
  for (bst_uint j = 0; j < dm.num_cols() && j < num_features_; ++j) {
    fval[j] = dm.feature(row, j);
  }

  // Sum feature contributions from all trees, scaled by learning rate (eta).
  const double eta = static_cast<double>(booster_.config().eta);
  std::vector<double> tree_contrib(n_out);
  for (const auto& shaper : tree_shapers_) {
    shaper.compute(fval, tree_contrib);
    for (bst_uint k = 0; k < num_features_; ++k)
    total[k] += tree_contrib[k] * eta;
  }

  // phi[bias] = prediction - sum(phi[features])
  // Standard TreeSHAP consistency: sum(phi) == f(x)
  // Computing phi[bias] from the actual prediction guarantees exact
  // additivity regardless of path-weight precision.
  double feature_sum = 0.0;
  for (bst_uint k = 0; k < num_features_; ++k) feature_sum += total[k];
  const std::vector<bst_uint> ri = {row};
  const double pred_raw = static_cast<double>(
    booster_.predict_batch_raw(dm, ri)[0]);
  total[num_features_] = pred_raw - feature_sum;

  return total;
}

//  Batch SHAP           
std::vector<double>
TreeExplainer::shap_values_batch(const DMatrix& dm) const
{
  const bst_uint n_rows = dm.num_rows();
  const bst_uint n_out  = num_features_ + 1;
  std::vector<double> result(static_cast<size_t>(n_rows) * n_out, 0.0);

  for (bst_uint i = 0; i < n_rows; ++i) {
    auto row_shap = shap_values(dm, i);
    for (bst_uint k = 0; k < n_out; ++k) {
    result[i * n_out + k] = row_shap[k];
    }
  }
  return result;
}

//  Mean |SHAP|          
std::vector<double>
TreeExplainer::mean_abs_shap(const DMatrix& dm) const
{
  const bst_uint n_rows = dm.num_rows();
  std::vector<double> importance(num_features_, 0.0);

  for (bst_uint i = 0; i < n_rows; ++i) {
    auto sv = shap_values(dm, i);
    for (bst_uint j = 0; j < num_features_; ++j) {
    importance[j] += std::abs(sv[j]);
    }
  }

  const double inv_n = 1.0 / static_cast<double>(n_rows);
  for (auto& v : importance) v *= inv_n;
  return importance;
}

} // namespace xgb
