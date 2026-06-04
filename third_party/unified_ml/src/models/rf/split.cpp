#include "models/rf/split.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rf {

// ---------------------------------------------------------------------------
// Impurity functions
// ---------------------------------------------------------------------------

double gini_impurity(const std::vector<double>& y,
       const std::vector<size_t>& indices,
       int n_classes) {
  if (indices.empty()) return 0.0;
  std::vector<int> counts(n_classes, 0);
  for (size_t i : indices) {
    int c = static_cast<int>(y[i]);
    if (c >= 0 && c < n_classes) ++counts[c];
  }
  double n = static_cast<double>(indices.size());
  double gini = 1.0;
  for (int c : counts) {
    double p = static_cast<double>(c) / n;
    gini -= p * p;
  }
  return gini;
}

double mse_impurity(const std::vector<double>& y,
        const std::vector<size_t>& indices) {
  return variance(y, indices);
}

double entropy_impurity(const std::vector<double>& y,
        const std::vector<size_t>& indices,
        int n_classes) {
  if (indices.empty()) return 0.0;
  std::vector<int> counts(n_classes, 0);
  for (size_t i : indices) {
    int c = static_cast<int>(y[i]);
    if (c >= 0 && c < n_classes) ++counts[c];
  }
  double n = static_cast<double>(indices.size());
  double ent = 0.0;
  for (int c : counts) {
    if (c == 0) continue;
    double p = static_cast<double>(c) / n;
    ent -= p * safe_log2(p);
  }
  return ent;
}

// ---------------------------------------------------------------------------
// Internal: partition indices based on feature threshold
// ---------------------------------------------------------------------------
static void partition(const Dataset& dataset,
        const std::vector<size_t>& indices,
        size_t feature_idx,
        double threshold,
        std::vector<size_t>& left,
        std::vector<size_t>& right) {
  left.clear();
  right.clear();
  for (size_t i : indices) {
    if (dataset.feature(i, feature_idx) <= threshold)
    left.push_back(i);
    else
    right.push_back(i);
  }
}

// ---------------------------------------------------------------------------
// SplitFinder — classification
// ---------------------------------------------------------------------------
SplitResult SplitFinder::best_threshold_classification(const Dataset& dataset,
                   const std::vector<size_t>& indices,
                   size_t feature_idx,
                   int n_classes) const {
  SplitResult best;
  best.feature_index = -1;

  // Sort indices by feature value
  std::vector<size_t> sorted = indices;
  std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
    return dataset.feature(a, feature_idx) < dataset.feature(b, feature_idx);
  });

  double n = static_cast<double>(indices.size());
  double parent_impurity = gini_impurity(dataset.y(), indices, n_classes);

  // Running counts for left side
  std::vector<int> left_counts(n_classes, 0);
  std::vector<int> right_counts(n_classes, 0);
  for (size_t i : indices) ++right_counts[static_cast<int>(dataset.y()[i])];

  size_t n_left  = 0;
  size_t n_right = indices.size();

  for (size_t k = 0; k + 1 < sorted.size(); ++k) {
    // Move sorted[k] from right to left
    int cls = static_cast<int>(dataset.y()[sorted[k]]);
    ++left_counts[cls];
    --right_counts[cls];
    ++n_left;
    --n_right;

    // Only consider threshold between distinct feature values
    double fk = dataset.feature(sorted[k],   feature_idx);
    double fk1  = dataset.feature(sorted[k + 1], feature_idx);
    if (fk == fk1) continue;

    // Skip if leaf size constraint violated
    if (static_cast<int>(n_left)  < params_.min_samples_leaf) continue;
    if (static_cast<int>(n_right) < params_.min_samples_leaf) continue;

    // Compute Gini for left
    double gini_left = 1.0;
    for (int c : left_counts) {
    double p = static_cast<double>(c) / static_cast<double>(n_left);
    gini_left -= p * p;
    }

    // Compute Gini for right
    double gini_right = 1.0;
    for (int c : right_counts) {
    double p = static_cast<double>(c) / static_cast<double>(n_right);
    gini_right -= p * p;
    }

    double wl = static_cast<double>(n_left)  / n;
    double wr = static_cast<double>(n_right) / n;
    double gain = parent_impurity - (wl * gini_left + wr * gini_right);

    if (gain > best.gain) {
    best.gain     = gain;
    best.feature_index  = static_cast<int>(feature_idx);
    best.threshold  = 0.5 * (fk + fk1);
    best.impurity_left  = gini_left;
    best.impurity_right = gini_right;
    }
  }
  return best;
}

SplitResult SplitFinder::best_split_classification(const Dataset& dataset,
                   const std::vector<size_t>& indices,
                   const std::vector<size_t>& feature_subset,
                   int n_classes) const {
  SplitResult global_best;
  for (size_t feat : feature_subset) {
    SplitResult candidate = best_threshold_classification(dataset, indices, feat, n_classes);
    if (candidate.valid() && candidate.gain > global_best.gain) {
    global_best = std::move(candidate);
    }
  }
  // Populate left/right index vectors for the winner
  if (global_best.valid()) {
    partition(dataset, indices,
      static_cast<size_t>(global_best.feature_index),
      global_best.threshold,
      global_best.left_indices,
      global_best.right_indices);
  }
  return global_best;
}

// ---------------------------------------------------------------------------
// SplitFinder — regression
// ---------------------------------------------------------------------------
SplitResult SplitFinder::best_threshold_regression(const Dataset& dataset,
                   const std::vector<size_t>& indices,
                   size_t feature_idx) const {
  SplitResult best;
  best.feature_index = -1;

  std::vector<size_t> sorted = indices;
  std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
    return dataset.feature(a, feature_idx) < dataset.feature(b, feature_idx);
  });

  double n = static_cast<double>(indices.size());
  double parent_mean = mean(dataset.y(), indices);

  // Running sums for incremental MSE
  double sum_left  = 0.0;
  double sum_right = 0.0;
  for (size_t i : indices) sum_right += dataset.y()[i];

  double sum_sq_left  = 0.0;
  double sum_sq_right = 0.0;
  for (size_t i : indices) {
    double v = dataset.y()[i];
    sum_sq_right += v * v;
  }

  size_t n_left  = 0;
  size_t n_right = indices.size();
  double parent_mse = sum_sq_right / n - parent_mean * parent_mean;

  for (size_t k = 0; k + 1 < sorted.size(); ++k) {
    double v = dataset.y()[sorted[k]];
    sum_left  += v;
    sum_sq_left += v * v;
    sum_right  -= v;
    sum_sq_right -= v * v;
    ++n_left;
    --n_right;

    double fk  = dataset.feature(sorted[k],   feature_idx);
    double fk1 = dataset.feature(sorted[k + 1], feature_idx);
    if (fk == fk1) continue;

    if (static_cast<int>(n_left)  < params_.min_samples_leaf) continue;
    if (static_cast<int>(n_right) < params_.min_samples_leaf) continue;

    double nl = static_cast<double>(n_left);
    double nr = static_cast<double>(n_right);

    // MSE = E[y^2] - (E[y])^2
    double mse_l = sum_sq_left  / nl - (sum_left  / nl) * (sum_left  / nl);
    double mse_r = sum_sq_right / nr - (sum_right / nr) * (sum_right / nr);
    mse_l = std::max(0.0, mse_l);
    mse_r = std::max(0.0, mse_r);

    double gain = parent_mse - (nl / n) * mse_l - (nr / n) * mse_r;

    if (gain > best.gain) {
    best.gain     = gain;
    best.feature_index  = static_cast<int>(feature_idx);
    best.threshold  = 0.5 * (fk + fk1);
    best.impurity_left  = mse_l;
    best.impurity_right = mse_r;
    }
  }
  return best;
}

SplitResult SplitFinder::best_split_regression(const Dataset& dataset,
                 const std::vector<size_t>& indices,
                 const std::vector<size_t>& feature_subset) const {
  SplitResult global_best;
  for (size_t feat : feature_subset) {
    SplitResult candidate = best_threshold_regression(dataset, indices, feat);
    if (candidate.valid() && candidate.gain > global_best.gain) {
    global_best = std::move(candidate);
    }
  }
  if (global_best.valid()) {
    partition(dataset, indices,
      static_cast<size_t>(global_best.feature_index),
      global_best.threshold,
      global_best.left_indices,
      global_best.right_indices);
  }
  return global_best;
}

} // namespace rf
