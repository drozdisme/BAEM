#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/objective/tweedie_regression.h
//
//  Tweedie regression objective: reg:tweedie
//
//  Useful for modelling semi-continuous non-negative targets
//  (e.g. insurance claim amounts, rainfall).
//
//  Raw prediction ŷ = exp(margin)  so the prediction is always positive.
//
//  Negative log-likelihood (per-sample loss):
//    ℓ(y, ŷ) = -y·exp((1-p)·margin)/(1-p)  +  exp((2-p)·margin)/(2-p)
//
//  where p = tweedie_variance_power ∈ (1, 2).
//
//  Gradient:
//    g = -y·exp((1-p)·margin) + exp((2-p)·margin)
//
//  Hessian:
//    h = -(1-p)·y·exp((1-p)·margin) + (2-p)·exp((2-p)·margin)
//
//  Both gradient and Hessian are computed from the raw margin score (log ŷ).
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/objective/loss_function.hpp"
#include <string>
#include <vector>

namespace xgb {

//                                       
//  TweedieRegressionObjective
//                                       
class TweedieRegressionObjective : public LossFunction {
public:
    // p : tweedie_variance_power ∈ (1, 2)
    explicit TweedieRegressionObjective(bst_float p = 1.5f);

    // Compute (g_i, h_i) for each sample.
    // scores : raw margins (log ŷ)
    // labels : non-negative target values
    void compute_gradients(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels,
        std::vector<GradientPair>&    out) const override;

    // ŷ = exp(margin)  →  transform is exp
    bst_float transform(bst_float margin) const override {
        return std::exp(margin);
    }

    // Scalar Tweedie deviance for logging.
    bst_float compute_loss(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels) const override;

    // Initial prediction = log(mean(y)) to avoid explosive gradients.
    bst_float base_score(const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "reg:tweedie"; }

    bst_float variance_power() const { return p_; }

private:
    bst_float p_;   // tweedie_variance_power
};

} // namespace xgb
