#include "models/xgboost/objective/squared_error.hpp"
#include <cmath>
#include <numeric>

namespace xgb {

// gi = ŷi - yi,  hi = 1
void SquaredError::compute_gradients(
  const std::vector<bst_float>& scores,
  const std::vector<bst_float>& labels,
  std::vector<GradientPair>& out) const
{
  out.resize(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    out[i].grad = scores[i] - labels[i];
    out[i].hess = 1.0f;
  }
}

bst_float SquaredError::compute_loss(
  const std::vector<bst_float>& scores,
  const std::vector<bst_float>& labels) const
{
  double loss = 0.0;
  for (size_t i = 0; i < scores.size(); ++i) {
    double diff = scores[i] - labels[i];
    loss += diff * diff;
  }
  return static_cast<bst_float>(std::sqrt(loss / scores.size())); // RMSE
}

bst_float SquaredError::base_score(
  const std::vector<bst_float>& labels) const
{
  if (labels.empty()) return 0.5f;
  double s = 0.0;
  for (bst_float v : labels) s += v;
  return static_cast<bst_float>(s / labels.size());
}

} // namespace xgb
