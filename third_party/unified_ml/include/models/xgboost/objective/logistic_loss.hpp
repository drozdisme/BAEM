#pragma once
#include "models/xgboost/objective/loss_function.hpp"
#include <cmath>

namespace xgb {

//                        
//  LogisticLoss  (binary:logistic)
//  l(ŷ, y) = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]
//  where σ(x) = 1 / (1 + e^-x)
//
//  gi = σ(ŷi) - yi
//  hi = σ(ŷi) · (1 - σ(ŷi))    (bounded from below at min_hess)
//                        
class LogisticLoss : public LossFunction {
public:
    explicit LogisticLoss(bst_float min_hess = 1e-16f)
        : min_hess_(min_hess) {}

    void compute_gradients(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels,
        std::vector<GradientPair>&    out) const override;

    // Raw margin → probability
    bst_float transform(bst_float margin) const override {
        return 1.f / (1.f + std::exp(-margin));
    }

    bst_float compute_loss(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels) const override;

    bst_float base_score(
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "binary:logistic"; }

private:
    bst_float min_hess_;

    static bst_float sigmoid(bst_float x) {
        return 1.f / (1.f + std::exp(-x));
    }
};

} // namespace xgb
