#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/booster/feature_importance.h
//
//  Feature 7: Three feature importance measures.
//
//  gain   — total improvement in the loss function brought by splits on
//            this feature (sum of split_gain across all nodes).
//            This is the original XGBoost "weight" in importance_type=gain.
//
//  weight  — number of times this feature appears in any split across the
//             ensemble (frequency / count).
//
//  cover   — average hessian (second derivative) of training samples that
//             pass through nodes split on this feature:
//               Cover_j = (1/N_j) Σ H_j
//             where N_j = number of nodes splitting on j,
//                   H_j = sum_hess at each such node.
//
//  All three are computed in O(T · depth) from the stored node statistics.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/tree/decision_tree.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace xgb {

//                                       
//  ImportanceType
//                                       
enum class ImportanceType {
    kGain,    // total split gain (default)
    kWeight,  // split frequency (count)
    kCover    // average hessian coverage
};

ImportanceType parse_importance_type(const std::string& s);

//                                       
//  FeatureImportance
//  Accumulates all three importance metrics across an ensemble of trees.
//                                       
class FeatureImportance {
public:
    explicit FeatureImportance(bst_uint num_features);

    // Add one tree's contribution.
    void accumulate(const DecisionTree& tree);

    // Retrieve the requested importance type.
    // Returns a normalised vector of length num_features_.
    // normalise=true: divide by the sum so values sum to 1.
    std::vector<bst_float> get(ImportanceType type,
                                bool normalise = false) const;

    std::vector<bst_float> get(const std::string& type_str,
                                bool normalise = false) const {
        return get(parse_importance_type(type_str), normalise);
    }

    bst_uint num_features() const { return num_features_; }

    // Clear all accumulators (allows reuse).
    void reset();

private:
    bst_uint num_features_;

    std::vector<double> gain_;    // sum of split gains per feature
    std::vector<bst_uint> weight_;  // split count per feature
    std::vector<double> cover_sum_; // sum of node hessians per feature
    std::vector<bst_uint> cover_cnt_; // node count per feature (for averaging)

    static std::vector<bst_float> normalise_vec(
        const std::vector<double>& raw);
};

} // namespace xgb
