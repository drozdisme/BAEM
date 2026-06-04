#pragma once
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include <memory>
#include <vector>

namespace xgb {

//                        
//  Predictor
//  Wraps a trained GradientBooster.
//  Handles:
//    - raw margin prediction
//    - probability prediction (post-transform)
//    - leaf index prediction (future: SHAP)
//                        
class Predictor {
public:
    explicit Predictor(std::shared_ptr<GradientBooster> booster);

    // Raw scores (margin)            
    std::vector<bst_float> predict_raw(const DMatrix& dm) const;

    // Probability (after sigmoid / softmax)   
    std::vector<bst_float> predict_proba(const DMatrix& dm) const;

    // For regression: same as raw        
    std::vector<bst_float> predict(const DMatrix& dm) const;

    // Predict using only first n_trees trees  
    // Useful for analysing contribution of each round.
    std::vector<bst_float> predict_ntree_limit(
        const DMatrix& dm,
        bst_uint n_trees) const;

    // Feature importance aggregated over all trees
    // Returns vector of length n_features
    std::vector<bst_float> feature_importance() const;

    // Getters                  
    const GradientBooster& booster() const { return *booster_; }

private:
    std::shared_ptr<GradientBooster> booster_;
    std::unique_ptr<LossFunction> loss_fn_;
};

} // namespace xgb
