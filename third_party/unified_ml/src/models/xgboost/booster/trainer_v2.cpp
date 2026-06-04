// ═══════════════════════════════════════════════════════════════════════════
//  src/booster/trainer_v2.cpp
//
//  Extended trainer with typed callback system.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/booster/trainer_v2.hpp"
#include "models/xgboost/metric/metric.hpp"
#include "models/xgboost/objective/loss_function.hpp"
#include <iostream>
#include <stdexcept>

namespace xgb {

//  Constructor           
TrainerV2::TrainerV2(const Config& cfg)
  : cfg_(cfg)
{
  metric_ = Metric::create(cfg_.booster().eval_metric);
}

//  Eval sets            
void TrainerV2::set_eval_set(std::shared_ptr<DMatrix> dm,
          const std::string& name)
{
  eval_sets_.emplace_back(std::move(dm), name);
}

//  Callback registration          
void TrainerV2::add_callback(std::unique_ptr<Callback> cb)
{
  callbacks_.push_back(std::move(cb));
}

//  Hook dispatch          
void TrainerV2::fire_train_begin(CallbackContext& ctx)
{
  for (auto& cb : callbacks_) cb->on_train_begin(ctx);
}

void TrainerV2::fire_iteration_begin(CallbackContext& ctx)
{
  for (auto& cb : callbacks_) cb->on_iteration_begin(ctx);
}

void TrainerV2::fire_iteration_end(CallbackContext& ctx)
{
  for (auto& cb : callbacks_) cb->on_iteration_end(ctx);
}

void TrainerV2::fire_train_end(CallbackContext& ctx)
{
  for (auto& cb : callbacks_) cb->on_train_end(ctx);
}

//  Evaluate all eval sets         
void TrainerV2::evaluate_all(const GradientBooster& booster,
          bst_int round,
          const std::shared_ptr<DMatrix>& train_dm)
{
  // Helper lambda: evaluate one DMatrix
  auto eval_one = [&](const DMatrix& dm, const std::string& ds_name) {
    const bst_uint n = dm.num_rows();
    std::vector<bst_uint> all_rows(n);
    for (bst_uint i = 0; i < n; ++i) all_rows[i] = i;

    auto raw_preds = booster.predict_batch_raw(dm, all_rows);

    // Apply objective transform to get probabilities
    auto loss_fn = LossFunction::create(cfg_.booster().objective, cfg_.booster().num_class);
    auto probs = loss_fn->transform_batch(raw_preds);

    const bst_float val = metric_->evaluate(probs, dm.labels());

    history_.results.push_back({ds_name, metric_->name(), val, round});
  };

  // Training set (always evaluated)
  if (train_dm) eval_one(*train_dm, "train");

  // Additional eval sets
  for (auto& [dm, name] : eval_sets_) {
    eval_one(*dm, name);
  }
}

//  Main train loop           
std::shared_ptr<GradientBooster>
TrainerV2::train(const std::shared_ptr<DMatrix>& train_dm)
{
  if (!train_dm) {
    throw std::invalid_argument("TrainerV2::train: train_dm is null");
  }

  const bst_int num_round = cfg_.booster().num_round;

  auto booster = std::make_shared<GradientBooster>(cfg_.booster());

  const bst_uint n = train_dm->num_rows();

  // For multiclass the score buffer must be flat [n * num_class].
  // Allocating only n entries causes OOB writes in boost_one_round
  // at index r*nc+c → SIGABRT.
  const bool is_multiclass = (cfg_.booster().objective == "multi:softmax" ||
           cfg_.booster().objective == "multi:softprob" ||
           cfg_.booster().objective == "softmax");
  const bst_int nc = is_multiclass
         ? std::max(cfg_.booster().num_class, bst_int(2))
         : 1;

  // Initialise scores at booster's base_score — identical to predict_batch_raw,
  // so training and prediction are always consistent.
  std::vector<bst_float> scores(static_cast<size_t>(n) * nc,
             booster->base_score());

  // Build shared CallbackContext       
  CallbackContext ctx;
  ctx.num_rounds = num_round;
  ctx.history  = &history_;
  ctx.booster  = booster.get();

  fire_train_begin(ctx);

  for (bst_int round = 0; round < num_round; ++round) {
    ctx.current_round = round;

    fire_iteration_begin(ctx);

    // Boost one round         
    const bst_float train_loss = booster->boost_one_round(*train_dm, scores);
    (void)train_loss; // stored in TrainHistory via evaluate_all

    // Evaluate          
    evaluate_all(*booster, round, train_dm);

    // Legacy callback         
    if (legacy_cb_) (*legacy_cb_)(round, history_);

    fire_iteration_end(ctx);

    // Early stopping check        
    if (ctx.stop_training) break;
  }

  fire_train_end(ctx);

  // Best-iteration rollback         
  // If EarlyStopping halted training, trim the booster back to the best
  // iteration so that predict() / score() use the optimal model, not the
  // last one (which may have over-fitted).
  if (ctx.stop_training && ctx.best_iteration >= 0) {
    const bst_uint keep = static_cast<bst_uint>(ctx.best_iteration + 1);
    booster->trim_trees(keep);
    std::cout << "[EarlyStopping] Rolled back to best iteration "
      << ctx.best_iteration << "  ("
      << booster->num_trees() << " trees kept)\n";
  }

  return booster;
}

} // namespace xgb
