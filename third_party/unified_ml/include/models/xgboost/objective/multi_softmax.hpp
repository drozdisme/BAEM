#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/objective/multi_softmax.h
//
//  Multiclass classification objective: multi:softmax
//
//  For C classes the booster maintains C score vectors simultaneously.
//  Each boosting round builds C trees — one per class.
//
//  Softmax probability for class c of sample i:
//    p_{ic} = exp(s_{ic}) / Σ_{c'} exp(s_{ic'})
//
//  Gradient and Hessian (same formulation as XGBoost paper Appendix):
//    g_{ic} = p_{ic} - 1[y_i == c]
//    h_{ic} = p_{ic} * (1 - p_{ic})
//
//  The flat gradient buffer is laid out as:
//    index  i*num_class + c  → GradientPair for sample i, class c
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/objective/loss_function.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace xgb {

//                                       
//  MultiSoftmaxObjective
//
//  Implements the multi:softmax / multi:softprob objective.
//  This class operates on the flat multi-class gradient buffer rather than
//  the standard single-class GradientPair vector.
//                                       
class MultiSoftmaxObjective {
public:
    explicit MultiSoftmaxObjective(bst_int num_class);

    // Compute multi-class gradients.
    // scores : flat buffer [n_samples * num_class_]  (raw margins per class)
    // labels : integer class labels in [0, num_class_)
    // out    : flat buffer [n_samples * num_class_]  (GradientPair per class)
    void compute_gradients(
        const std::vector<bst_float>&  scores,
        const std::vector<bst_float>&  labels,
        std::vector<GradientPair>&     out) const;

    // Apply softmax to a flat score buffer, returns probability vector.
    // in/out : flat [n_samples * num_class_]
    std::vector<bst_float> softmax_transform(
        const std::vector<bst_float>& scores) const;

    // Return the winning class for each sample.
    std::vector<bst_int> predict_class(
        const std::vector<bst_float>& scores) const;

    // Scalar cross-entropy loss for logging.
    bst_float compute_loss(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels) const;

    bst_int num_class() const { return num_class_; }

    std::string name() const { return "multi:softmax"; }

private:
    bst_int num_class_;

    // Numerically stable softmax over a C-element window.
    // writes probabilities into probs[offset .. offset+C)
    void softmax_window(const bst_float* scores_begin,
                        bst_float*       probs_begin) const;
};

} // namespace xgb
