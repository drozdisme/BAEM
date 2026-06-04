#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/booster/trainer_v2.h
//
//  Extended Trainer with typed callback architecture (Feature 6).
//
//  Drop-in replacement for trainer.h — preserves full backward compatibility
//  while adding:
//    • add_callback(unique_ptr<Callback>)
//    • Typed hook dispatch: on_train_begin / on_iteration_end / on_train_end
//    • CallbackContext propagation through all hooks
//    • stop_training signal from EarlyStopping
//
//  Usage:
//    TrainerV2 trainer(config);
//    trainer.set_eval_set(val_dm, "val");
//    trainer.add_callback(std::make_unique<EarlyStopping>("val","auc",10,true));
//    trainer.add_callback(std::make_unique<ModelCheckpoint>("ckpt_{round}.json","val","auc",true));
//    trainer.add_callback(std::make_unique<LoggingCallback>(5));
//    auto booster = trainer.train(train_dm);
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/booster/trainer.hpp"            // re-use EvalResult, TrainHistory
#include "models/xgboost/metric/metric.hpp"
#include "models/xgboost/callback/callback.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <optional>

namespace xgb {

//                                       
//  TrainerV2
//  Extends Trainer with a typed, pluggable callback system.
//                                       
class TrainerV2 {
public:
    explicit TrainerV2(const Config& cfg);

    //   Eval sets (logged every round)                   
    void set_eval_set(std::shared_ptr<DMatrix> dm, const std::string& name);

    //   Callback registration                        
    void add_callback(std::unique_ptr<Callback> cb);

    //   Legacy single-function callback (backward compat.)         
    using RoundCallback =
        std::function<void(bst_int round, const TrainHistory&)>;
    void set_callback(RoundCallback cb) { legacy_cb_ = std::move(cb); }

    //   Train                                
    std::shared_ptr<GradientBooster> train(
        const std::shared_ptr<DMatrix>& train_dm);

    //   History                               
    const TrainHistory& history() const { return history_; }

private:
    Config cfg_;
    std::vector<std::pair<std::shared_ptr<DMatrix>, std::string>> eval_sets_;
    std::unique_ptr<Metric>   metric_;
    TrainHistory              history_;
    std::optional<RoundCallback> legacy_cb_;

    // Typed callbacks in registration order
    std::vector<std::unique_ptr<Callback>> callbacks_;

    //   Hook dispatch helpers                        
    void fire_train_begin    (CallbackContext& ctx);
    void fire_iteration_begin(CallbackContext& ctx);
    void fire_iteration_end  (CallbackContext& ctx);
    void fire_train_end      (CallbackContext& ctx);

    //   Evaluation                             
    void evaluate_all(const GradientBooster& booster,
                      bst_int round,
                      const std::shared_ptr<DMatrix>& train_dm);
};

} // namespace xgb
