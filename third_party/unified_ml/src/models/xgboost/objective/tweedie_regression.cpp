// ═══════════════════════════════════════════════════════════════════════════
//  src/objective/tweedie_regression.cpp
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/objective/tweedie_regression.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace xgb {

TweedieRegressionObjective::TweedieRegressionObjective(bst_float p)
  : p_(p)
{
  if (p_ <= 1.f || p_ >= 2.f) {
    throw std::invalid_argument(
    "TweedieRegression: tweedie_variance_power must be in (1, 2). "
    "Got: " + std::to_string(p_));
  }
}

//  Gradient and Hessian          
//
//  Raw margin s = log(ŷ).
//
//  g_i = -y_i · exp((1-p)·s_i) + exp((2-p)·s_i)
//  h_i = -(1-p)·y_i · exp((1-p)·s_i) + (2-p) · exp((2-p)·s_i)
//
//  h is always positive for p ∈ (1,2) and y ≥ 0, but we clip to 1e-6.
//
void TweedieRegressionObjective::compute_gradients(
  const std::vector<bst_float>& scores,
  const std::vector<bst_float>& labels,
  std::vector<GradientPair>&  out) const
{
  const size_t n = labels.size();
  assert(scores.size() == n);
  out.resize(n);

  const bst_float one_minus_p = 1.f - p_;
  const bst_float two_minus_p = 2.f - p_;

  for (size_t i = 0; i < n; ++i) {
    const bst_float s = scores[i]; // raw margin (log ŷ)
    const bst_float y = labels[i];

    // Clamp margins to avoid overflow in exp
    const bst_float s1 = std::min(s * one_minus_p, 30.f);
    const bst_float s2 = std::min(s * two_minus_p, 30.f);

    const bst_float e1 = std::exp(s1); // exp((1-p)·s)
    const bst_float e2 = std::exp(s2); // exp((2-p)·s)

    out[i].grad = -y * e1 + e2;
    out[i].hess = std::max(-one_minus_p * y * e1 + two_minus_p * e2,
           1e-6f);
  }
}

//  Loss (Tweedie deviance)        
bst_float TweedieRegressionObjective::compute_loss(
  const std::vector<bst_float>& scores,
  const std::vector<bst_float>& labels) const
{
  const size_t n = labels.size();
  bst_float loss = 0.f;

  const bst_float one_minus_p = 1.f - p_;
  const bst_float two_minus_p = 2.f - p_;

  for (size_t i = 0; i < n; ++i) {
    const bst_float s = scores[i];
    const bst_float y = labels[i];

    // -y·exp((1-p)·s)/(1-p) + exp((2-p)·s)/(2-p)
    loss += -y * std::exp(s * one_minus_p) / one_minus_p
       + std::exp(s * two_minus_p) / two_minus_p;
  }
  return loss / static_cast<bst_float>(n);
}

//  Base score = log(mean(y))         
bst_float TweedieRegressionObjective::base_score(
  const std::vector<bst_float>& labels) const
{
  if (labels.empty()) return 0.f;

  const double sum = std::accumulate(labels.begin(), labels.end(), 0.0);
  const double mean = sum / static_cast<double>(labels.size());

  // log of mean; clamp mean to > 0
  return static_cast<bst_float>(std::log(std::max(mean, 1e-6)));
}

} // namespace xgb
