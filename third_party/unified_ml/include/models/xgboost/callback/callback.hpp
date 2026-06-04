#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/callback/callback.h
//
//  Typed callback architecture for the training loop.
//
//  Hook points:
//    on_train_begin     — called once before the first round
//    on_iteration_begin — called at the start of each round
//    on_iteration_end   — called after each round (with metrics)
//    on_train_end       — called once after the last round
//
//  Design:  Each callback is an independent object registered with the
//           Trainer.  The Trainer holds a vector<unique_ptr<Callback>> and
//           invokes hooks in registration order.  Callbacks communicate
//           back through CallbackContext (e.g. early-stopping signals the
//           trainer to halt).
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <limits>

namespace xgb {

// Forward declarations
class GradientBooster;
struct TrainHistory;

//                                       
//  CallbackContext
//  Shared mutable context passed to every hook.
//  Callbacks use it to read state and signal the trainer.
//                                       
struct CallbackContext {
    bst_int   current_round   {0};
    bst_int   num_rounds      {0};
    bool      stop_training   {false};   // set to true by EarlyStopping
    bst_int   best_iteration  {-1};      // set by EarlyStopping on stop
    const TrainHistory*   history  {nullptr};
    GradientBooster*      booster  {nullptr};   // non-owning ptr
};

//                                       
//  Callback — abstract base
//                                       
class Callback {
public:
    virtual ~Callback() = default;

    virtual void on_train_begin     (CallbackContext& /*ctx*/) {}
    virtual void on_iteration_begin (CallbackContext& /*ctx*/) {}
    virtual void on_iteration_end   (CallbackContext& /*ctx*/) {}
    virtual void on_train_end       (CallbackContext& /*ctx*/) {}

    virtual std::string name() const = 0;
};

//                                       
//  EarlyStopping  (Feature 2)
//
//  Stops training when the watched metric on a specific dataset does not
//  improve for `rounds` consecutive iterations.
//
//  The metric direction (higher/lower is better) is taken from the Metric
//  object itself so no manual flag is needed.
//                                       
class EarlyStopping : public Callback {
public:
    // dataset_name : name passed to Trainer::set_eval_set()
    // metric_name  : e.g. "rmse", "auc", "accuracy"
    // rounds       : patience — number of non-improving rounds before halt
    // higher_is_better : true for accuracy/AUC, false for rmse/logloss
    EarlyStopping(std::string dataset_name,
                  std::string metric_name,
                  bst_int     rounds,
                  bool        higher_is_better = false);

    void on_train_begin  (CallbackContext& ctx) override;
    void on_iteration_end(CallbackContext& ctx) override;
    void on_train_end    (CallbackContext& ctx) override;

    std::string name() const override { return "EarlyStopping"; }

    bst_int best_iteration()    const { return best_iteration_; }
    bst_float best_score()      const { return best_score_;     }

private:
    std::string dataset_name_;
    std::string metric_name_;
    bst_int     rounds_;
    bool        higher_is_better_;

    bst_int   best_iteration_  {0};
    bst_int   no_improve_count_{0};
    bst_float best_score_;

    bool is_better(bst_float candidate, bst_float best) const;
};

//                                       
//  ModelCheckpoint  (Feature 2, companion to EarlyStopping)
//
//  Saves the model to disk whenever the watched metric improves.
//  Works independently of EarlyStopping — both can be registered.
//                                       
class ModelCheckpoint : public Callback {
public:
    // path_template : e.g. "model_{round}.json" — {round} is substituted
    // save_best_only: if true, only saves when metric improves
    ModelCheckpoint(std::string   path_template,
                    std::string   dataset_name,
                    std::string   metric_name,
                    bool          higher_is_better = false,
                    bool          save_best_only   = true);

    void on_iteration_end(CallbackContext& ctx) override;

    std::string name() const override { return "ModelCheckpoint"; }

private:
    std::string path_template_;
    std::string dataset_name_;
    std::string metric_name_;
    bool        higher_is_better_;
    bool        save_best_only_;
    bst_float   best_score_;

    std::string make_path(bst_int round) const;
    bool        is_better(bst_float candidate, bst_float best) const;
};

//                                       
//  LearningRateScheduler
//
//  Multiplies the booster's eta by a user-supplied schedule function.
//  schedule_fn(round) → eta for that round.
//                                       
class LearningRateScheduler : public Callback {
public:
    using ScheduleFn = std::function<bst_float(bst_int round)>;

    // Built-in schedules
    static ScheduleFn cosine_annealing(bst_float eta_max,
                                       bst_float eta_min,
                                       bst_int   T_max);

    static ScheduleFn step_decay(bst_float initial_lr,
                                 bst_float drop_factor,
                                 bst_int   step_size);

    explicit LearningRateScheduler(ScheduleFn fn)
        : fn_(std::move(fn)) {}

    void on_iteration_begin(CallbackContext& ctx) override;

    std::string name() const override { return "LearningRateScheduler"; }

private:
    ScheduleFn fn_;
};

//                                       
//  LoggingCallback
//
//  Prints a summary line after each round.
//  Replaces the trainer's built-in print_last() logic.
//                                       
class LoggingCallback : public Callback {
public:
    explicit LoggingCallback(bst_int print_every = 1)
        : print_every_(print_every) {}

    void on_iteration_end(CallbackContext& ctx) override;

    std::string name() const override { return "LoggingCallback"; }

private:
    bst_int print_every_;
};

} // namespace xgb
