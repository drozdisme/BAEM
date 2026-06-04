#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/objective/softmax_loss_adapter.h
//
//  LossFunction adapter for MultiSoftmaxObjective.
//
//  MultiSoftmaxObjective is a standalone class that does NOT inherit from
//  LossFunction (because it operates on a flat n*C score buffer rather than
//  the standard n-element vector).  This adapter bridges the gap so that
//  GradientBooster::boost_one_round() can dispatch multi:softmax through the
//  same LossFunction interface as all other objectives.
//
//  Key differences vs standard LossFunction:
//    • compute_gradients() ignores `scores.size() == n` and instead expects
//      `scores.size() == n * num_class`.  This is satisfied by the test and
//      by GradientBooster when cfg.num_class > 1.
//    • compute_loss() delegates to MultiSoftmaxObjective::compute_loss().
//    • base_score() returns 0.5 / num_class (balanced initialisation).
//    • transform() is a no-op (argmax is done outside LossFunction in the
//      multiclass path).
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/objective/loss_function.hpp"
#include "models/xgboost/objective/multi_softmax.hpp"
#include <memory>
#include <string>
#include <vector>

namespace xgb {

class SoftmaxLossAdapter : public LossFunction {
public:
    explicit SoftmaxLossAdapter(bst_int num_class)
        : obj_(num_class) {}

    // Delegate gradient computation to MultiSoftmaxObjective.
    // scores layout: [n * num_class]  (flat multi-class buffer)
    void compute_gradients(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels,
        std::vector<GradientPair>&    out) const override
    {
        obj_.compute_gradients(scores, labels, out);
    }

    // Raw margin pass-through (no transform for softmax — caller does argmax).
    bst_float transform(bst_float margin) const override { return margin; }

    // Batch transform: apply softmax row-wise so predict() returns probabilities.
    std::vector<bst_float> transform_batch(
        const std::vector<bst_float>& margins) const override
    {
        return obj_.softmax_transform(margins);
    }

    // Cross-entropy loss for logging.
    bst_float compute_loss(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels) const override
    {
        return obj_.compute_loss(scores, labels);
    }

    // Balanced initialisation: start all class scores at the same value.
    bst_float base_score(const std::vector<bst_float>& /*labels*/) const override
    {
        return 0.f;
    }

    std::string name() const override { return "multi:softmax"; }

    bst_int num_class() const { return obj_.num_class(); }

private:
    MultiSoftmaxObjective obj_;
};

} // namespace xgb
