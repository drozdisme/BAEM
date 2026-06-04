#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/metric/auc.h
//
//  AUC-ROC metric for binary classification.
//
//  Exact definition:
//    AUC = (1 / n+ n-) Σ_{i:y=1} Σ_{j:y=0} 1[p̂_i > p̂_j]
//
//  Efficient O(n log n) implementation via rank statistics:
//    1. Sort samples by predicted score descending.
//    2. Track cumulative positives to compute trapezoidal AUC.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/metric/metric.hpp"
#include <vector>
#include <string>

namespace xgb {

//                                       
//  AUCMetric  — Area Under the ROC Curve
//                                       
class AUCMetric : public Metric {
public:
    // Evaluate AUC-ROC.
    // predictions : probability scores in [0, 1]
    // labels      : binary {0, 1}
    bst_float evaluate(
        const std::vector<bst_float>& predictions,
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "auc"; }

    // AUC: higher is better.
    bool higher_is_better() const override { return true; }
};

} // namespace xgb
