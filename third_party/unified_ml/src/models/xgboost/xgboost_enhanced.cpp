// =============================================================================
//  src/xgboost_enhanced.cpp  —  XGBModel implementation
// =============================================================================
#include "models/xgboost/xgboost_enhanced.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <iomanip>

namespace fs  = std::filesystem;
using Clock   = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

namespace xgb {

// =============================================================================
//  ShapResult helpers
// =============================================================================
std::vector<std::pair<std::string, double>>
ShapResult::top_features(bst_uint row, bst_uint k) const
{
  if (row >= n_rows)
    throw std::out_of_range("ShapResult: row index out of range");

  std::vector<std::pair<std::string, double>> out;
  out.reserve(n_features);
  for (bst_uint j = 0; j < n_features; ++j)
    out.emplace_back(names.size() > j ? names[j] : "f" + std::to_string(j),
         values[row][j]);

  std::sort(out.begin(), out.end(),
      [](const auto& a, const auto& b) {
      return std::fabs(a.second) > std::fabs(b.second);
      });
  if (k > 0 && k < out.size()) out.resize(k);
  return out;
}

// =============================================================================
//  Metrics::print
// =============================================================================
void Metrics::print() const
{
  std::cout << "\n  Metrics (" << task << ")    \n";
  if (task == "regression") {
    std::cout << std::fixed << std::setprecision(5)
      << "  RMSE   : " << rmse   << "\n"
      << "  MAE  : " << mae  << "\n"
      << "  R²   : " << r2   << "\n";
  } else if (task == "binary") {
    std::cout << std::fixed << std::setprecision(5)
      << "  AUC  : " << auc  << "\n"
      << "  Accuracy : " << accuracy << "\n"
      << "  LogLoss  : " << logloss  << "\n";
  } else {
    std::cout << std::fixed << std::setprecision(5)
      << "  Accuracy : " << accuracy << "\n";
  }
  std::cout << std::fixed << std::setprecision(1)
      << "  Train ms : " << train_ms << "\n"
      << "  Infer ms : " << infer_ms << "\n\n";
}

// =============================================================================
//  XGBModel — Construction
// =============================================================================
XGBModel::XGBModel(const std::string& task)
  : task_(task), history_(std::make_shared<TrainHistory>())
{
  Logger::instance().set_level(LogLevel::WARNING);  // quiet by default
}

// =============================================================================
//  Fluent setters
// =============================================================================
XGBModel& XGBModel::task     (const std::string& t) { task_ = t; return *this; }
XGBModel& XGBModel::n_estimators (int n) { n_estimators_  = n;  return *this; }
XGBModel& XGBModel::learning_rate  (float l) { learning_rate_ = l;  return *this; }
XGBModel& XGBModel::max_depth  (int d) { max_depth_   = d;  return *this; }
XGBModel& XGBModel::subsample  (float s) { subsample_   = s;  return *this; }
XGBModel& XGBModel::colsample  (float c) { colsample_   = c;  return *this; }
XGBModel& XGBModel::lambda   (float l) { lambda_    = l;  return *this; }
XGBModel& XGBModel::gamma    (float g) { gamma_   = g;  return *this; }
XGBModel& XGBModel::grow_policy  (const std::string& p) { grow_policy_ = p; return *this; }
XGBModel& XGBModel::num_class  (int n) { num_class_   = n;  return *this; }
XGBModel& XGBModel::feature_names  (const std::vector<std::string>& names) {
  feature_names_ = names; return *this;
}

XGBModel& XGBModel::early_stopping(int patience, bool higher) {
  early_stopping_rounds_ = patience;
  es_higher_is_better_ = higher;
  return *this;
}
XGBModel& XGBModel::checkpoint(const std::string& path) {
  checkpoint_path_ = path; return *this;
}
XGBModel& XGBModel::verbose(int every) {
  verbose_every_ = every; return *this;
}
XGBModel& XGBModel::lr_cosine(float eta_max, float eta_min, int t_max) {
  use_lr_cosine_ = true;
  lr_cos_max_ = eta_max; lr_cos_min_ = eta_min; lr_cos_tmax_ = t_max;
  return *this;
}
XGBModel& XGBModel::lr_step(float initial, float drop, int step_size) {
  use_lr_step_   = true;
  lr_step_initial_ = initial;
  lr_step_drop_  = drop;
  lr_step_size_  = step_size;
  return *this;
}

// =============================================================================
//  Internal helpers
// =============================================================================
std::string XGBModel::metric_name() const {
  if (task_ == "regression" || task_ == "tweedie") return "rmse";
  if (task_ == "binary")          return "auc";
  return "accuracy";
}
bool XGBModel::higher_is_better() const {
  const auto m = metric_name();
  return (m == "auc" || m == "accuracy");
}

Config XGBModel::build_config() const {
  Config cfg;
  std::string obj = "reg:squarederror";
  if  (task_ == "binary")   obj = "binary:logistic";
  else if (task_ == "multiclass") obj = "multi:softmax";
  else if (task_ == "tweedie")  obj = "reg:tweedie";
  else if (task_ == "ranking")  obj = "rank:pairwise";

  cfg.set_objective(obj)
   .set_num_round(n_estimators_)
   .set_eta(learning_rate_)
   .set_max_depth(max_depth_)
   .set_subsample(subsample_)
   .set_lambda(lambda_)
   .set_gamma(gamma_);

  cfg.booster().eval_metric  = metric_name();
  cfg.tree().grow_policy   = grow_policy_;

  if (task_ == "multiclass")
    cfg.booster().num_class = num_class_;

  return cfg;
}

Metrics XGBModel::compute_metrics(const std::vector<bst_float>& y_true,
             const std::vector<bst_float>& y_pred) const
{
  Metrics m;
  m.task  = task_;
  m.train_ms  = train_ms_;
  m.infer_ms  = infer_ms_;
  const size_t n = y_true.size();

  if (task_ == "regression" || task_ == "tweedie") {
    double rmse = 0, mae = 0, y_mean = 0;
    for (size_t i = 0; i < n; i++) {
    double e = y_true[i] - y_pred[i];
    rmse += e * e; mae += std::fabs(e); y_mean += y_true[i];
    }
    rmse = std::sqrt(rmse / n); mae /= n; y_mean /= n;
    double ss_res = 0, ss_tot = 0;
    for (size_t i = 0; i < n; i++) {
    ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    ss_tot += (y_true[i] - y_mean)  * (y_true[i] - y_mean);
    }
    m.rmse = rmse; m.mae = mae;
    m.r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
  }
  else if (task_ == "binary") {
    AUCMetric auc_m;
    m.auc = auc_m.evaluate(y_pred, y_true);
    int ok = 0;
    double ll = 0;
    for (size_t i = 0; i < n; i++) {
    ok += ((y_pred[i] > 0.5f) == (y_true[i] > 0.5f));
    double p = std::max(1e-7, std::min(1 - 1e-7, (double)y_pred[i]));
    ll -= y_true[i] * std::log(p) + (1 - y_true[i]) * std::log(1 - p);
    }
    m.accuracy = static_cast<double>(ok) / n;
    m.logloss  = ll / n;
  }
  else {
    int ok = 0;
    for (size_t i = 0; i < n; i++)
    ok += (std::round(y_pred[i]) == std::round(y_true[i]));
    m.accuracy = static_cast<double>(ok) / n;
  }
  return m;
}

// =============================================================================
//  Core fit() — takes two DMatrix pointers
// =============================================================================
XGBModel& XGBModel::fit(std::shared_ptr<DMatrix> train_dm,
         std::shared_ptr<DMatrix> val_dm)
{
  if (!feature_names_.empty())
    train_dm->set_feature_names(feature_names_);

  Config cfg = build_config();
  TrainerV2 trainer(cfg);

  if (val_dm)
    trainer.set_eval_set(val_dm, "val");

  // Callbacks           
  if (verbose_every_ > 0)
    trainer.add_callback(std::make_unique<LoggingCallback>(verbose_every_));

  if (early_stopping_rounds_ > 0 && val_dm) {
    bool hib = es_higher_is_better_ ? es_higher_is_better_
              : higher_is_better();
    trainer.add_callback(std::make_unique<EarlyStopping>(
    "val", metric_name(), early_stopping_rounds_, hib));
  }

  if (!checkpoint_path_.empty() && val_dm) {
    trainer.add_callback(std::make_unique<ModelCheckpoint>(
    checkpoint_path_, "val", metric_name(),
    higher_is_better(), /*save_best_only=*/true));
  }

  if (use_lr_cosine_)
    trainer.add_callback(std::make_unique<LearningRateScheduler>(
    LearningRateScheduler::cosine_annealing(
      lr_cos_max_, lr_cos_min_, lr_cos_tmax_)));

  if (use_lr_step_)
    trainer.add_callback(std::make_unique<LearningRateScheduler>(
    LearningRateScheduler::step_decay(
      lr_step_initial_, lr_step_drop_, lr_step_size_)));

  // Train            
  auto t0 = Clock::now();
  booster_ = trainer.train(train_dm);
  train_ms_ = Ms(Clock::now() - t0).count();

  *history_   = trainer.history();
  last_train_dm_ = train_dm;
  last_val_dm_ = val_dm;

  return *this;
}

// Convenience overloads          
XGBModel& XGBModel::fit(const std::vector<std::vector<bst_float>>& X,
         const std::vector<bst_float>& y)
{
  return fit(DMatrix::from_dense(X, y), nullptr);
}

XGBModel& XGBModel::fit(const core::MatrixView& X,
         const std::vector<bst_float>& y)
{
  return fit(DMatrix::from_matrix_view(X, y), nullptr);
}

XGBModel& XGBModel::fit(const std::vector<std::vector<bst_float>>& X_train,
         const std::vector<bst_float>&     y_train,
         const std::vector<std::vector<bst_float>>& X_val,
         const std::vector<bst_float>&     y_val)
{
  return fit(DMatrix::from_dense(X_train, y_train),
     DMatrix::from_dense(X_val, y_val));
}

XGBModel& XGBModel::fit(const core::MatrixView& X_train,
         const std::vector<bst_float>& y_train,
         const core::MatrixView& X_val,
         const std::vector<bst_float>& y_val)
{
  return fit(DMatrix::from_matrix_view(X_train, y_train),
     DMatrix::from_matrix_view(X_val, y_val));
}

XGBModel& XGBModel::fit_csv(const std::string& train_path,
          const std::string& val_path,
          int      label_col)
{
  CSVLoadOptions opts;
  opts.label_column = label_col;
  auto train_dm = DataLoader::load_csv(train_path, opts);
  std::shared_ptr<DMatrix> val_dm;
  if (!val_path.empty())
    val_dm = DataLoader::load_csv(val_path, opts);
  return fit(train_dm, val_dm);
}

// =============================================================================
//  predict()
// =============================================================================
std::vector<bst_float> XGBModel::predict(std::shared_ptr<DMatrix> dm) const
{
  if (!booster_) throw std::runtime_error("XGBModel: call fit() before predict()");
  Predictor pred(booster_);
  auto t0 = Clock::now();
  auto out = pred.predict(*dm);
  const_cast<XGBModel*>(this)->infer_ms_ = Ms(Clock::now() - t0).count();
  return out;
}

std::vector<bst_float>
XGBModel::predict(const std::vector<std::vector<bst_float>>& X) const
{
  return predict(DMatrix::from_dense(X, {}));
}

std::vector<bst_float>
XGBModel::predict(const core::MatrixView& X) const
{
  return predict(DMatrix::from_matrix_view(X, {}));
}

std::vector<bst_float>
XGBModel::predict_csv(const std::string& path, int label_col) const
{
  CSVLoadOptions opts; opts.label_column = label_col;
  return predict(DataLoader::load_csv(path, opts));
}

// =============================================================================
//  score()
// =============================================================================
Metrics XGBModel::score(std::shared_ptr<DMatrix> dm) const
{
  auto preds = predict(dm);
  return compute_metrics(dm->labels(), preds);
}

Metrics XGBModel::score(const std::vector<std::vector<bst_float>>& X,
         const std::vector<bst_float>& y) const
{
  return score(DMatrix::from_dense(X, y));
}

Metrics XGBModel::score(const core::MatrixView& X,
         const std::vector<bst_float>& y) const
{
  return score(DMatrix::from_matrix_view(X, y));
}

// =============================================================================
//  feature_importances()
// =============================================================================
std::vector<std::pair<std::string, bst_float>>
XGBModel::feature_importances(const std::string& type) const
{
  if (!booster_) throw std::runtime_error("XGBModel: call fit() first");

  const bst_uint nf = booster_->trees().empty()
        ? static_cast<bst_uint>(feature_names_.size())
        : last_train_dm_ ? last_train_dm_->num_features() : 0;

  FeatureImportance fi(nf);
  for (const auto& t : booster_->trees()) fi.accumulate(*t);

  auto raw = fi.get(type, /*normalise=*/true);

  // Build name→value pairs
  std::vector<std::pair<std::string, bst_float>> result;
  result.reserve(nf);
  for (bst_uint j = 0; j < nf; ++j) {
    std::string nm = (j < feature_names_.size())
         ? feature_names_[j]
         : "f" + std::to_string(j);
    result.emplace_back(nm, raw[j]);
  }
  // Sort descending by importance
  std::sort(result.begin(), result.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });
  return result;
}

// =============================================================================
//  explain()  — TreeSHAP
// =============================================================================
ShapResult XGBModel::explain(std::shared_ptr<DMatrix> dm,
           bst_uint max_rows) const
{
  if (!booster_) throw std::runtime_error("XGBModel: call fit() first");
  const bst_uint nf = dm->num_features();
  TreeExplainer explainer(*booster_, nf);

  const bst_uint n = std::min(dm->num_rows(), max_rows);
  auto flat = explainer.shap_values_batch(*dm);  // [n x (nf+1)] row-major

  ShapResult out;
  out.n_rows   = n;
  out.n_features = nf;
  out.values.resize(n, std::vector<double>(nf));
  out.bias.resize(n);

  for (bst_uint i = 0; i < n; ++i) {
    for (bst_uint j = 0; j < nf; ++j)
    out.values[i][j] = flat[i * (nf + 1) + j];
    out.bias[i] = flat[i * (nf + 1) + nf];
  }

  // Feature names
  if (!feature_names_.empty())
    out.names = feature_names_;
  else if (dm->feature_names().size() == nf)
    out.names = dm->feature_names();
  else {
    out.names.resize(nf);
    for (bst_uint j = 0; j < nf; ++j)
    out.names[j] = "f" + std::to_string(j);
  }
  return out;
}

ShapResult XGBModel::explain(const std::vector<std::vector<bst_float>>& X,
           bst_uint max_rows) const
{
  return explain(DMatrix::from_dense(X, {}), max_rows);
}

ShapResult XGBModel::explain(const core::MatrixView& X,
           bst_uint max_rows) const
{
  return explain(DMatrix::from_matrix_view(X, {}), max_rows);
}

// =============================================================================
//  Persistence
// =============================================================================
void XGBModel::save(const std::string& path) const
{
  if (!booster_) throw std::runtime_error("XGBModel: nothing to save");
  booster_->save_model(path);
}

void XGBModel::load(const std::string& path)
{
  if (!booster_) booster_ = std::make_shared<GradientBooster>(build_config().booster());
  booster_->load_model(path);
}

void XGBModel::save_binary(const std::string& path) const
{
  if (!booster_) throw std::runtime_error("XGBModel: nothing to save");
  booster_->save_model_binary(path);
}

void XGBModel::load_binary(const std::string& path)
{
  if (!booster_) booster_ = std::make_shared<GradientBooster>(build_config().booster());
  booster_->load_model_binary(path);
}

// =============================================================================
//  Accessors
// =============================================================================
bst_uint XGBModel::num_trees()  const { return booster_ ? booster_->num_trees() : 0; }
bst_uint XGBModel::num_features() const {
  return last_train_dm_ ? last_train_dm_->num_features()
          : static_cast<bst_uint>(feature_names_.size());
}
const TrainHistory& XGBModel::history() const { return *history_; }

void XGBModel::print_summary() const
{
  if (!booster_) { std::cout << "XGBModel: not fitted\n"; return; }

  std::cout << "\n╔═══════════════════════════════════════╗\n"
      << "║    XGBModel summary     ║\n"
      << "╚═══════════════════════════════════════╝\n"
      << "  Task    : " << task_ << "\n"
      << "  Trees   : " << num_trees() << "\n"
      << "  Features  : " << num_features() << "\n"
      << std::fixed << std::setprecision(1)
      << "  Train time  : " << train_ms_ << " ms\n"
      << "  Infer time  : " << infer_ms_ << " ms\n\n";

  if (!feature_names_.empty()) {
    auto fi = feature_importances("gain");
    std::cout << "  Top feature importances (gain):\n";
    const bst_uint show = std::min((bst_uint)5, (bst_uint)fi.size());
    for (bst_uint i = 0; i < show; ++i)
    std::cout << "  " << std::setw(16) << std::left << fi[i].first
        << std::right << std::setprecision(3) << fi[i].second * 100 << "%\n";
    std::cout << "\n";
  }
}

// =============================================================================
//  Dashboard export helpers
// =============================================================================
std::string XGBModel::jstr(const std::string& s)  { return "\"" + s + "\""; }
std::string XGBModel::jkvs(const std::string& k, const std::string& v)
  { return "  " + jstr(k) + ": " + jstr(v); }
std::string XGBModel::jkvn(const std::string& k, double v) {
  std::ostringstream ss; ss << std::setprecision(6) << v;
  return "  " + jstr(k) + ": " + ss.str();
}
std::string XGBModel::jkv(const std::string& k, const std::string& raw)
  { return "  " + jstr(k) + ": " + raw; }
std::string XGBModel::jarr(const std::vector<std::string>& v) {
  std::string r = "[";
  for (size_t i = 0; i < v.size(); ++i) { if (i) r += ", "; r += jstr(v[i]); }
  return r + "]";
}
std::string XGBModel::op(const std::string& dir, const std::string& f)
  { return dir + "/" + f; }

// =============================================================================
//  export_dashboard()
// =============================================================================
void XGBModel::export_dashboard(const std::string& output_dir,
           std::shared_ptr<DMatrix> test_dm) const
{
  if (!booster_) throw std::runtime_error("XGBModel: call fit() first");

  fs::create_directories(output_dir);

  // Use last val DMatrix as test if not provided
  auto dm = test_dm ? test_dm : last_val_dm_;
  if (!dm) throw std::runtime_error("XGBModel: provide test_dm for dashboard export");

  auto preds = predict(dm);
  write_dashboard_files(output_dir, dm, preds);
  std::cout << "Dashboard written to: " << output_dir << "/\n";
}

void XGBModel::write_dashboard_files(const std::string& dir,
              std::shared_ptr<DMatrix> test_dm,
              const std::vector<bst_float>& preds) const
{
  const auto& y_true = test_dm->labels();
  const bst_uint nf  = test_dm->num_features();
  const bst_uint n = test_dm->num_rows();

  // Feature names (effective)
  std::vector<std::string> fnames;
  if (!feature_names_.empty() && feature_names_.size() == nf)
    fnames = feature_names_;
  else if (test_dm->feature_names().size() == nf)
    fnames = test_dm->feature_names();
  else {
    fnames.resize(nf);
    for (bst_uint j = 0; j < nf; ++j) fnames[j] = "f" + std::to_string(j);
  }

  // run_info.json          
  {
    std::ofstream f(op(dir, "run_info.json"));
    f << "{\n"
    << jkvs("task", task_)        << ",\n"
    << jkv ("feature_names", jarr(fnames))    << ",\n"
    << jkvn("n_estimators",  (double)n_estimators_) << ",\n"
    << jkvn("learning_rate", learning_rate_)  << ",\n"
    << jkvn("max_depth",   max_depth_)    << ",\n"
    << jkvn("subsample",   subsample_)    << ",\n"
    << jkvs("grow_policy", grow_policy_)    << ",\n"
    << jkvn("train_ms",  train_ms_)     << ",\n"
    << jkvn("infer_ms",  infer_ms_)     << "\n}\n";
  }

  // predictions.csv         
  {
    std::ofstream f(op(dir, "predictions.csv"));
    f << "sample_id,y_true,y_pred\n";
    for (bst_uint i = 0; i < n; ++i)
    f << i << "," << std::setprecision(8)
      << (i < y_true.size() ? y_true[i] : 0.f) << ","
      << preds[i] << "\n";
  }

  // feature_importance.csv         
  {
    FeatureImportance fi(nf);
    for (const auto& t : booster_->trees()) fi.accumulate(*t);
    auto gr = fi.get("gain", false), gp = fi.get("gain", true);
    auto wr = fi.get("weight", false), wp = fi.get("weight", true);
    auto cr = fi.get("cover",  false), cp = fi.get("cover",  true);

    std::ofstream f(op(dir, "feature_importance.csv"));
    f << "feature,gain,gain_pct,weight,weight_pct,cover,cover_pct\n";
    for (bst_uint j = 0; j < nf; ++j)
    f << fnames[j] << "," << std::setprecision(8)
      << gr[j] << "," << gp[j] << ","
      << wr[j] << "," << wp[j] << ","
      << cr[j] << "," << cp[j] << "\n";
  }

  // test_features.csv          
  {
    std::ofstream f(op(dir, "test_features.csv"));
    for (bst_uint j = 0; j < nf; ++j) { if (j) f << ","; f << fnames[j]; }
    f << ",y_true\n";
    for (bst_uint i = 0; i < n; ++i) {
    for (bst_uint j = 0; j < nf; ++j) {
      if (j) f << ",";
      f << std::setprecision(8) << test_dm->feature(i, j);
    }
    float lbl = y_true.size() > i ? y_true[i] : 0.f;
    f << "," << lbl << "\n";
    }
  }

  // shap_values.csv         
  {
    TreeExplainer explainer(*booster_, nf);
    const bst_uint shap_n = std::min(n, (bst_uint)300);
    auto phi = explainer.shap_values_batch(*test_dm);

    std::ofstream f(op(dir, "shap_values.csv"));
    for (bst_uint j = 0; j < nf; ++j) { if (j) f << ","; f << fnames[j]; }
    f << ",bias\n";
    for (bst_uint i = 0; i < shap_n; ++i) {
    for (bst_uint j = 0; j <= nf; ++j) {
      if (j) f << ",";
      f << std::setprecision(8) << phi[i * (nf + 1) + j];
    }
    f << "\n";
    }
  }

  // metrics.json          
  {
    Metrics m = compute_metrics(y_true, preds);
    std::ofstream f(op(dir, "metrics.json"));
    f << "{\n" << jkvs("task", task_) << ",\n"
    << jkvn("train_ms", train_ms_)  << ",\n"
    << jkvn("infer_ms", infer_ms_)  << ",\n";
    if (task_ == "regression" || task_ == "tweedie")
    f << jkvn("rmse", m.rmse) << ",\n"
      << jkvn("mae",  m.mae)  << ",\n"
      << jkvn("r2", m.r2) << "\n";
    else if (task_ == "binary")
    f << jkvn("auc",  m.auc)  << ",\n"
      << jkvn("accuracy", m.accuracy) << ",\n"
      << jkvn("logloss",  m.logloss)  << "\n";
    else
    f << jkvn("accuracy", m.accuracy) << "\n";
    f << "}\n";
  }

  // train_history.csv          
  history_->save_csv(op(dir, "train_history.csv"));

  // model.json           
  booster_->save_model(op(dir, "model.json"));

  // trees.txt          
  {
    std::ofstream tf(op(dir, "trees.txt"));
    tf << booster_->dump_model_text();
  }
}

} // namespace xgb
