#include "models/xgboost/booster/trainer.hpp"
#include "models/xgboost/objective/loss_function.hpp"
#include "models/xgboost/utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace xgb {

// TrainHistory          

std::vector<bst_float> TrainHistory::values_for(
  const std::string& dataset_name,
  const std::string& metric_name) const
{
  std::vector<bst_float> out;
  for (const auto& r : results)
    if (r.dataset_name == dataset_name && r.metric_name == metric_name)
    out.push_back(r.value);
  return out;
}

void TrainHistory::print_last(bst_int round) const {
  std::cout << "[" << std::setw(4) << round << "]";
  for (const auto& r : results)
    if (r.round == round)
    std::cout << "  " << r.dataset_name << "-" << r.metric_name
        << ": " << std::fixed << std::setprecision(6) << r.value;
  std::cout << "\n";
}

void TrainHistory::save_csv(const std::string& path) const {
  std::ofstream f(path);
  if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
  f << "round,dataset,metric,value\n";
  for (const auto& r : results)
    f << r.round << "," << r.dataset_name << ","
    << r.metric_name << "," << r.value << "\n";
}

// Trainer           

Trainer::Trainer(const Config& cfg)
  : cfg_(cfg)
  , metric_(Metric::create(cfg.booster().eval_metric))
{
}

void Trainer::set_eval_set(
  std::shared_ptr<DMatrix> dm, const std::string& name)
{
  eval_sets_.push_back({std::move(dm), name});
}

std::shared_ptr<GradientBooster> Trainer::train(
  const std::shared_ptr<DMatrix>& train_dm)
{
  if (!train_dm->has_labels())
    throw std::runtime_error("Trainer: training data has no labels");

  const bst_uint n = train_dm->num_rows();
  const auto& bcfg = cfg_.booster();

  auto booster = std::make_shared<GradientBooster>(bcfg);
  auto loss_fn = LossFunction::create(bcfg.objective, bcfg.num_class);
  bst_float init_score = loss_fn->base_score(train_dm->labels());

  // Sync booster's base_score_ with the data-derived init_score so that
  // predict_batch_raw() starts from the same value as training.
  booster->set_base_score(init_score);

  // For multiclass the score buffer is flat [n * num_class].
  const bool multiclass = (bcfg.objective == "multi:softmax" ||
           bcfg.objective == "multi:softprob" ||
           bcfg.objective == "softmax");
  const bst_int nc = multiclass ? std::max(bcfg.num_class, bst_int(2)) : 1;

  // Working score vector (accumulated raw margins)
  std::vector<bst_float> scores(static_cast<size_t>(n) * nc, init_score);

  XGB_LOG_INFO("Training XGBoost — " + std::to_string(bcfg.num_round)
       + " rounds, objective=" + bcfg.objective);
  cfg_.print();

  for (bst_int round = 0; round < bcfg.num_round; ++round) {
    booster->boost_one_round(*train_dm, scores);
    evaluate_all(*booster, round, train_dm);

    if (round % 10 == 0 || round == bcfg.num_round - 1)
    history_.print_last(round);

    if (callback_)
    (*callback_)(round, history_);
  }

  XGB_LOG_INFO("Training complete — " +
       std::to_string(booster->num_trees()) + " trees built");
  return booster;
}

void Trainer::evaluate_all(
  const GradientBooster& booster,
  bst_int round,
  const std::shared_ptr<DMatrix>& train_dm)
{
  auto loss_fn = LossFunction::create(cfg_.booster().objective, cfg_.booster().num_class);

  // Evaluate on training set
  {
    bst_uint n = train_dm->num_rows();
    std::vector<bst_uint> all_rows(n);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    auto raw = booster.predict_batch_raw(*train_dm, all_rows);
    auto preds = loss_fn->transform_batch(raw);
    bst_float val = metric_->evaluate(preds, train_dm->labels());
    history_.results.push_back({"train", metric_->name(), val, round});
  }

  // Evaluate on eval sets
  for (const auto& [dm, name] : eval_sets_) {
    bst_uint n = dm->num_rows();
    std::vector<bst_uint> all_rows(n);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    auto raw = booster.predict_batch_raw(*dm, all_rows);
    auto preds = loss_fn->transform_batch(raw);
    bst_float val = metric_->evaluate(preds, dm->labels());
    history_.results.push_back({name, metric_->name(), val, round});
  }
}

} // namespace xgb
