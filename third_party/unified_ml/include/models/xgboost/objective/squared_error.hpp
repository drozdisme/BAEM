#pragma once
#include "models/xgboost/objective/loss_function.hpp"

namespace xgb {

//                        
//  SquaredError  (reg:squarederror)
//  l(ŷ, y) = ½(ŷ - y)²
//  gi = ŷi - yi
//  hi = 1
//                        
class SquaredError : public LossFunction {
public:
    void compute_gradients(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels,
        std::vector<GradientPair>&    out) const override;

    bst_float compute_loss(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels) const override;

    bst_float base_score(
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "reg:squarederror"; }
};

} // namespace xgb
