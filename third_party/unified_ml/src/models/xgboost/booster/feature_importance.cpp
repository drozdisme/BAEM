// ═══════════════════════════════════════════════════════════════════════════
//  src/booster/feature_importance.cpp
//
//  Gain, Weight (frequency), and Cover importance metrics.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/booster/feature_importance.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace xgb {

//    parse_importance_type                          
ImportanceType parse_importance_type(const std::string& s)
{
    if (s == "gain"   || s == "total_gain") return ImportanceType::kGain;
    if (s == "weight" || s == "frequency" ) return ImportanceType::kWeight;
    if (s == "cover"  || s == "total_cover") return ImportanceType::kCover;
    throw std::invalid_argument(
        "Unknown importance type: '" + s + "'. "
        "Expected: 'gain', 'weight', or 'cover'.");
}

//    Constructor                               
FeatureImportance::FeatureImportance(bst_uint num_features)
    : num_features_(num_features)
    , gain_     (num_features, 0.0)
    , weight_   (num_features, 0)
    , cover_sum_(num_features, 0.0)
    , cover_cnt_(num_features, 0)
{}

void FeatureImportance::reset()
{
    std::fill(gain_.begin(),      gain_.end(),      0.0);
    std::fill(weight_.begin(),    weight_.end(),    0);
    std::fill(cover_sum_.begin(), cover_sum_.end(), 0.0);
    std::fill(cover_cnt_.begin(), cover_cnt_.end(), 0);
}

//    accumulate                               
//
//  Walk all internal (non-leaf) nodes of one tree.
//  For each split node with feature j:
//    gain_[j]      += node.split_gain
//    weight_[j]    += 1
//    cover_sum_[j] += node.sum_hess
//    cover_cnt_[j] += 1
//
void FeatureImportance::accumulate(const DecisionTree& tree)
{
    for (const auto& node : tree.nodes()) {
        if (node.is_leaf || !node.is_valid) continue;

        const bst_uint fi = node.feature_idx;
        if (fi >= num_features_) continue;

        gain_     [fi] += static_cast<double>(node.split_gain);
        weight_   [fi] += 1;
        cover_sum_[fi] += node.sum_hess;
        cover_cnt_[fi] += 1;
    }
}

//    get                                   
std::vector<bst_float>
FeatureImportance::get(ImportanceType type, bool normalise) const
{
    std::vector<double> raw(num_features_, 0.0);

    switch (type) {
        case ImportanceType::kGain:
            raw = gain_;
            break;

        case ImportanceType::kWeight:
            for (bst_uint j = 0; j < num_features_; ++j) {
                raw[j] = static_cast<double>(weight_[j]);
            }
            break;

        case ImportanceType::kCover:
            // Cover_j = sum_hess_j / count_j   (average hessian per split)
            for (bst_uint j = 0; j < num_features_; ++j) {
                raw[j] = (cover_cnt_[j] > 0)
                         ? cover_sum_[j] / static_cast<double>(cover_cnt_[j])
                         : 0.0;
            }
            break;
    }

    if (normalise) {
        return normalise_vec(raw);
    }

    // Convert to bst_float without normalising
    std::vector<bst_float> result(num_features_);
    for (bst_uint j = 0; j < num_features_; ++j) {
        result[j] = static_cast<bst_float>(raw[j]);
    }
    return result;
}

//    normalise_vec                              
std::vector<bst_float>
FeatureImportance::normalise_vec(const std::vector<double>& raw)
{
    const double total = std::accumulate(raw.begin(), raw.end(), 0.0);
    std::vector<bst_float> result(raw.size());

    if (total > 0.0) {
        for (size_t j = 0; j < raw.size(); ++j) {
            result[j] = static_cast<bst_float>(raw[j] / total);
        }
    }
    return result;
}

} // namespace xgb
