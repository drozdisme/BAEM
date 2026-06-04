#include "models/xgboost/predictor/predictor.hpp"
#include "models/xgboost/objective/loss_function.hpp"
#include <numeric>
#include <stdexcept>

namespace xgb {

Predictor::Predictor(std::shared_ptr<GradientBooster> booster)
  : booster_(std::move(booster))
  , loss_fn_(LossFunction::create(booster_->config().objective, booster_->config().num_class))
{
}

std::vector<bst_float> Predictor::predict_raw(const DMatrix& dm) const {
  bst_uint n = dm.num_rows();
  std::vector<bst_uint> all_rows(n);
  std::iota(all_rows.begin(), all_rows.end(), 0);
  return booster_->predict_batch_raw(dm, all_rows);
}

std::vector<bst_float> Predictor::predict_proba(const DMatrix& dm) const {
  auto raw = predict_raw(dm);
  return loss_fn_->transform_batch(raw);
}

std::vector<bst_float> Predictor::predict(const DMatrix& dm) const {
  // For regression: identity transform (returns raw margin).
  // For classification: returns probability.
  return predict_proba(dm);
}

std::vector<bst_float> Predictor::predict_ntree_limit(
  const DMatrix& dm, bst_uint n_trees) const
{
  bst_uint n = dm.num_rows();
  const auto& cfg = booster_->config();
  bst_uint limit = std::min(n_trees, booster_->num_trees());

  std::vector<bst_float> scores(n, booster_->base_score());
  for (bst_uint t = 0; t < limit; ++t) {
    const auto& tree = booster_->trees()[t];
    for (bst_uint r = 0; r < n; ++r)
    scores[r] += cfg.eta * tree->predict(dm, r);
  }
  return loss_fn_->transform_batch(scores);
}

std::vector<bst_float> Predictor::feature_importance() const {
  return booster_->feature_importance(
    /* n_features — we don't know it without a DMatrix, so return 0-padded */
    1024); // reasonable upper bound; caller can trim
}

} // namespace xgb
