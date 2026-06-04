#include "models/xgboost/metric/metric.hpp"
#include "models/xgboost/metric/auc.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace xgb {

// Factory           
// Supported metric names:
// rmse  — root mean squared error
// mae   — mean absolute error
// accuracy  — binary accuracy (threshold = 0.5)
// logloss — binary cross-entropy
// auc   — AUC-ROC (binary classification)

std::unique_ptr<Metric> Metric::create(const std::string& name) {
  if (name == "rmse")   return std::make_unique<RMSE>();
  if (name == "mae")  return std::make_unique<MAE>();
  if (name == "accuracy") return std::make_unique<BinaryAccuracy>();
  if (name == "logloss")  return std::make_unique<LogLoss>();
  if (name == "auc")  return std::make_unique<AUCMetric>();
  // Default fallback
  return std::make_unique<RMSE>();
}

// RMSE            

bst_float RMSE::evaluate(
  const std::vector<bst_float>& preds,
  const std::vector<bst_float>& labels) const
{
  if (preds.size() != labels.size())
    throw std::runtime_error("RMSE: size mismatch");
  double sum = 0.0;
  for (size_t i = 0; i < preds.size(); ++i) {
    double d = preds[i] - labels[i];
    sum += d * d;
  }
  return static_cast<bst_float>(std::sqrt(sum / preds.size()));
}

// MAE             

bst_float MAE::evaluate(
  const std::vector<bst_float>& preds,
  const std::vector<bst_float>& labels) const
{
  double sum = 0.0;
  for (size_t i = 0; i < preds.size(); ++i)
    sum += std::abs(preds[i] - labels[i]);
  return static_cast<bst_float>(sum / preds.size());
}

// BinaryAccuracy           

bst_float BinaryAccuracy::evaluate(
  const std::vector<bst_float>& preds,
  const std::vector<bst_float>& labels) const
{
  bst_uint correct = 0;
  for (size_t i = 0; i < preds.size(); ++i) {
    int pred_class  = (preds[i] >= threshold_) ? 1 : 0;
    int label_class = (labels[i] >= 0.5f)  ? 1 : 0;
    if (pred_class == label_class) ++correct;
  }
  return static_cast<bst_float>(correct) / static_cast<bst_float>(preds.size());
}

// LogLoss           

bst_float LogLoss::evaluate(
  const std::vector<bst_float>& preds,
  const std::vector<bst_float>& labels) const
{
  constexpr bst_float eps = 1e-7f;
  double loss = 0.0;
  for (size_t i = 0; i < preds.size(); ++i) {
    bst_float p = std::max(eps, std::min(1.f - eps, preds[i]));
    loss -= labels[i] * std::log(p)
      + (1.f - labels[i]) * std::log(1.f - p);
  }
  return static_cast<bst_float>(loss / preds.size());
}

} // namespace xgb
