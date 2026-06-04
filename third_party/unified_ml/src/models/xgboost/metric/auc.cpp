// ═══════════════════════════════════════════════════════════════════════════
//  src/metric/auc.cpp
//
//  AUC-ROC via rank statistics  O(n log n).
//
//  Algorithm (Wilcoxon-Mann-Whitney statistic):
//  Let n+ = number of positive labels
//  Let n- = number of negative labels
//  Sort all samples by score ascending.
//  R = sum of ranks (1-based) of positive samples in the sorted list.
//  AUC = (R - n+(n++1)/2) / (n+ * n-)
//
//  Ties are handled by averaging ranks (same as sklearn).
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/metric/auc.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace xgb {

bst_float AUCMetric::evaluate(
  const std::vector<bst_float>& predictions,
  const std::vector<bst_float>& labels) const
{
  const size_t n = predictions.size();
  if (n != labels.size()) {
    throw std::invalid_argument("AUCMetric: predictions and labels size mismatch");
  }
  if (n == 0) return 0.f;

  // Step 1: Create index array and sort by prediction score ascending  
  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    return predictions[a] < predictions[b];
  });

  // Step 2: Assign average ranks to handle ties     
  //  Tied predictions (same score) share the average of their ranks.
  std::vector<double> ranks(n);
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    // Find end of tie group
    while (j < n && predictions[idx[j]] == predictions[idx[i]]) ++j;
    // Average rank (1-based) for this group
    const double avg_rank = 1.0 + (static_cast<double>(i) + static_cast<double>(j) - 1.0) / 2.0;
    for (size_t k = i; k < j; ++k) {
    ranks[idx[k]] = avg_rank;
    }
    i = j;
  }

  // Step 3: Wilcoxon-Mann-Whitney statistic     
  double n_pos = 0.0, n_neg = 0.0;
  double rank_sum = 0.0; // sum of ranks for positive samples

  for (size_t s = 0; s < n; ++s) {
    if (labels[s] >= 0.5f) {   // positive sample
    ++n_pos;
    rank_sum += ranks[s];
    } else {
    ++n_neg;
    }
  }

  if (n_pos == 0.0 || n_neg == 0.0) {
    // Degenerate case: only one class present → AUC undefined, return 0.5
    return 0.5f;
  }

  // AUC = (R - n+(n++1)/2) / (n+ * n-)
  const double auc = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);

  // Clamp to [0, 1] to guard against floating-point edge cases
  return static_cast<bst_float>(std::max(0.0, std::min(1.0, auc)));
}

} // namespace xgb
