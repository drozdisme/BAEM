#pragma once
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <string>
#include <memory>

namespace xgb {

//                        
//  Metric  — abstract evaluation metric
//                        
class Metric {
public:
    virtual ~Metric() = default;

    // Evaluate predictions against labels    
    // predictions: post-transform values (probs / values)
    // labels:      ground truth
    virtual bst_float evaluate(
        const std::vector<bst_float>& predictions,
        const std::vector<bst_float>& labels) const = 0;

    virtual std::string name() const = 0;

    // Higher is better? (for early stopping)  
    virtual bool higher_is_better() const { return false; }

    // Factory                  
    static std::unique_ptr<Metric> create(const std::string& name);
};

//                        
//  RMSE  — Root Mean Squared Error
//  RMSE = sqrt(1/n Σ(ŷi - yi)²)
//                        
class RMSE : public Metric {
public:
    bst_float evaluate(
        const std::vector<bst_float>& predictions,
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "rmse"; }
};

//                        
//  MAE  — Mean Absolute Error
//                        
class MAE : public Metric {
public:
    bst_float evaluate(
        const std::vector<bst_float>& predictions,
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "mae"; }
};

//                        
//  BinaryAccuracy  — threshold = 0.5
//                        
class BinaryAccuracy : public Metric {
public:
    explicit BinaryAccuracy(bst_float threshold = 0.5f)
        : threshold_(threshold) {}

    bst_float evaluate(
        const std::vector<bst_float>& predictions,
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "accuracy"; }
    bool higher_is_better() const override { return true; }

private:
    bst_float threshold_;
};

//                        
//  LogLoss  — Binary cross-entropy
//                        
class LogLoss : public Metric {
public:
    bst_float evaluate(
        const std::vector<bst_float>& predictions,
        const std::vector<bst_float>& labels) const override;

    std::string name() const override { return "logloss"; }
};

} // namespace xgb
