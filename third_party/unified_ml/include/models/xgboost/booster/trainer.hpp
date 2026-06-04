#pragma once
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/metric/metric.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <optional>

namespace xgb {

//                        
//  EvalResult — one metric on one dataset
//                        
struct EvalResult {
    std::string dataset_name;
    std::string metric_name;
    bst_float   value;
    bst_int     round;
};

//                        
//  TrainHistory — collected over all rounds
//                        
struct TrainHistory {
    std::vector<EvalResult> results;

    // Filter by dataset / metric        
    std::vector<bst_float> values_for(
        const std::string& dataset_name,
        const std::string& metric_name) const;

    void print_last(bst_int round) const;
    void save_csv(const std::string& path) const;
};

//                        
//  Trainer
//  Orchestrates the full training loop.
//
//  Usage:
//    Trainer trainer(config);
//    trainer.set_eval_set(eval_dm, "validation");
//    auto booster = trainer.train(train_dm);
//                        
class Trainer {
public:
    using RoundCallback =
        std::function<void(bst_int round, const TrainHistory&)>;

    explicit Trainer(const Config& cfg);

    // Optional eval sets (logged each round)  
    void set_eval_set(std::shared_ptr<DMatrix> dm,
                      const std::string& name);

    // Optional callback (called after each round)
    void set_callback(RoundCallback cb) { callback_ = std::move(cb); }

    // Run num_round iterations          
    std::shared_ptr<GradientBooster> train(
        const std::shared_ptr<DMatrix>& train_dm);

    // Access history after training       
    const TrainHistory& history() const { return history_; }

private:
    Config cfg_;
    std::vector<std::pair<std::shared_ptr<DMatrix>, std::string>> eval_sets_;
    std::unique_ptr<Metric> metric_;
    TrainHistory history_;
    std::optional<RoundCallback> callback_;

    void evaluate_all(const GradientBooster& booster,
                      bst_int round,
                      const std::shared_ptr<DMatrix>& train_dm);
};

} // namespace xgb
