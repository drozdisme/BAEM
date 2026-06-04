#include "models/iforest/itree.hpp"
#include "models/iforest/math_utils.hpp"
#include "models/iforest/random_utils.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace iforest {

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void ITree::build(
  const std::vector<std::vector<double>>& data,
  const std::vector<std::size_t>&   indices,
  std::mt19937&         rng,
  int             max_height)
{
  root_ = build_recursive(data, indices, 0, max_height, rng);
}

void ITree::build(
  const core::MatrixView& data,
  const std::vector<std::size_t>& indices,
  std::mt19937& rng,
  int max_height)
{
  root_ = build_recursive(data, indices, 0, max_height, rng);
}

double ITree::path_length(const std::vector<double>& sample) const {
  if (!root_) {
    throw std::runtime_error(
    "ITree::path_length called before build().");
  }
  return path_length_recursive(root_.get(), sample, 0.0);
}

double ITree::path_length(const double* sample_ptr) const {
  if (!root_) {
    throw std::runtime_error(
    "ITree::path_length called before build().");
  }
  return path_length_recursive(root_.get(), sample_ptr, 0.0);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::unique_ptr<Node> ITree::build_recursive(
  const std::vector<std::vector<double>>& data,
  const std::vector<std::size_t>&   indices,
  int             current_depth,
  int             max_height,
  std::mt19937&         rng)
{
  auto node = std::make_unique<Node>();

  // Termination conditions        
  if (indices.size() <= 1 || current_depth >= max_height) {
    node->leaf_size = indices.size();
    return node;  // is_leaf() == true (feature_index remains -1)
  }

  const std::size_t n_features = data[0].size();

  // Shuffle feature indices so we try them in random order.
  std::vector<std::size_t> feature_order(n_features);
  std::iota(feature_order.begin(), feature_order.end(), std::size_t{0});
  std::shuffle(feature_order.begin(), feature_order.end(), rng);

  // Try each feature until we find a valid split     
  for (std::size_t feat : feature_order) {

    double mn =  std::numeric_limits<double>::max();
    double mx = -std::numeric_limits<double>::max();

    for (std::size_t idx : indices) {
    const double v = data[idx][feat];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    }

    if (mx <= mn) continue;  // All values identical — no split possible.

    const double split = random_uniform(mn, mx, rng);

    std::vector<std::size_t> left_idx, right_idx;
    left_idx.reserve(indices.size());
    right_idx.reserve(indices.size());

    for (std::size_t idx : indices) {
    (data[idx][feat] < split ? left_idx : right_idx).push_back(idx);
    }

    if (left_idx.empty() || right_idx.empty()) continue;

    // Valid split found         
    node->feature_index = static_cast<int>(feat);
    node->split_value = split;
    node->left  = build_recursive(data, left_idx,  current_depth + 1, max_height, rng);
    node->right = build_recursive(data, right_idx, current_depth + 1, max_height, rng);
    return node;
  }

  // All features were constant in this partition — create a leaf.
  node->leaf_size = indices.size();
  return node;
}

std::unique_ptr<Node> ITree::build_recursive(
  const core::MatrixView& data,
  const std::vector<std::size_t>& indices,
  int current_depth,
  int max_height,
  std::mt19937& rng)
{
  auto node = std::make_unique<Node>();

  if (indices.size() <= 1 || current_depth >= max_height) {
    node->leaf_size = indices.size();
    return node;
  }

  const std::size_t n_features = data.cols;
  std::vector<std::size_t> feature_order(n_features);
  std::iota(feature_order.begin(), feature_order.end(), std::size_t{0});
  std::shuffle(feature_order.begin(), feature_order.end(), rng);

  for (std::size_t feat : feature_order) {
    double mn =  std::numeric_limits<double>::max();
    double mx = -std::numeric_limits<double>::max();

    for (std::size_t idx : indices) {
      const double v = data.data[idx * data.row_stride + feat];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }

    if (mx <= mn) continue;

    const double split = random_uniform(mn, mx, rng);
    std::vector<std::size_t> left_idx, right_idx;
    left_idx.reserve(indices.size());
    right_idx.reserve(indices.size());

    for (std::size_t idx : indices) {
      ((data.data[idx * data.row_stride + feat] < split) ? left_idx : right_idx).push_back(idx);
    }

    if (left_idx.empty() || right_idx.empty()) continue;

    node->feature_index = static_cast<int>(feat);
    node->split_value = split;
    node->left  = build_recursive(data, left_idx,  current_depth + 1, max_height, rng);
    node->right = build_recursive(data, right_idx, current_depth + 1, max_height, rng);
    return node;
  }

  node->leaf_size = indices.size();
  return node;
}

double ITree::path_length_recursive(
  const Node*      node,
  const std::vector<double>& sample,
  double       current_depth) const
{
  if (node->is_leaf()) {
    return current_depth + c(node->leaf_size);
  }

  const std::size_t f = static_cast<std::size_t>(node->feature_index);
  if (sample[f] < node->split_value) {
    return path_length_recursive(node->left.get(),  sample, current_depth + 1.0);
  } else {
    return path_length_recursive(node->right.get(), sample, current_depth + 1.0);
  }
}

double ITree::path_length_recursive(
  const Node* node,
  const double* sample_ptr,
  double current_depth) const
{
  if (node->is_leaf()) {
    return current_depth + c(node->leaf_size);
  }

  const std::size_t f = static_cast<std::size_t>(node->feature_index);
  if (sample_ptr[f] < node->split_value) {
    return path_length_recursive(node->left.get(), sample_ptr, current_depth + 1.0);
  } else {
    return path_length_recursive(node->right.get(), sample_ptr, current_depth + 1.0);
  }
}

} // namespace iforest
