#pragma once
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <memory>
#include <string>

namespace xgb {

//                        
//  LossFunction  (objective function)
//  Abstract base for all loss functions.
//
//  Role:
//    1. Compute (gi, hi) gradient pairs for each sample.
//    2. Transform raw margin scores to probabilities.
//    3. Compute scalar loss for logging.
//
//  See Sec. 2.2: gi = ∂l/∂ŷ,  hi = ∂²l/∂ŷ²
//                        
class LossFunction {
public:
    virtual ~LossFunction() = default;

    // Compute gradient pairs for all samples  
    // scores = current raw predictions (margins)
    // labels = ground truth
    // out    = output vector (resized automatically)
    virtual void compute_gradients(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels,
        std::vector<GradientPair>&    out) const = 0;

    // Transform raw margin → probability    
    virtual bst_float transform(bst_float margin) const { return margin; }

    // Batch transform
    virtual std::vector<bst_float> transform_batch(
        const std::vector<bst_float>& margins) const;

    // Initial prediction (base score)      
    virtual bst_float base_score(const std::vector<bst_float>& labels) const;

    // Scalar loss (for logging only)      
    virtual bst_float compute_loss(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels) const = 0;

    virtual std::string name() const = 0;

    // Factory                  
    // num_class is used by multi:softmax; ignored by all other objectives.
    static std::unique_ptr<LossFunction> create(const std::string& name,
                                                 bst_int num_class = 2);
};

} // namespace xgb
