// =============================================================================
//  xgboost_enhanced.h  —  Single-header facade
//
//  Usage:
//  #include "models/xgboost/xgboost_enhanced.hpp"
//
//  xgb::XGBModel model;
//  model.task("regression").n_estimators(300).learning_rate(0.1f)
//     .max_depth(5).early_stopping(20).verbose(25);
//
//  model.fit(X_train, y_train, X_val, y_val);
//
//  auto preds  = model.predict(X_test);
//  auto rmse   = model.score(X_test, y_test);
//  auto importance = model.feature_importances();
//  auto shap   = model.explain(X_test);
//
//  model.save("model.json");
//  model.export_dashboard("output/");
//
//  Convenience aliases:
//  xgb::XGBRegressor — same as XGBModel, task preset to "regression"
//  xgb::XGBClassifier  — same as XGBModel, task preset to "binary"
// =============================================================================
#pragma once

// Low-level library headers (all public)       
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/data/data_loader.hpp"
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/booster/trainer.hpp"
#include "models/xgboost/booster/trainer_v2.hpp"
#include "models/xgboost/booster/feature_importance.hpp"
#include "models/xgboost/predictor/predictor.hpp"
#include "models/xgboost/metric/metric.hpp"
#include "models/xgboost/metric/auc.hpp"
#include "models/xgboost/callback/callback.hpp"
#include "models/xgboost/explain/tree_shap.hpp"
#include "models/xgboost/utils/logger.hpp"
#include "models/xgboost/utils/csv_reader.hpp"
#include "core/matrix_view.hpp"

// STL             
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace xgb {

// =============================================================================
//  ShapResult — output of XGBModel::explain()
//
//  Values: [n_rows × n_features] matrix
//  Bias:   [n_rows] bias term (expected model output, phi_0)
//  Names:  feature names
// =============================================================================
struct ShapResult {
  std::vector<std::vector<double>> values; // [row][feature]
  std::vector<double>      bias;   // [row]
  std::vector<std::string>   names;  // feature names
  bst_uint         n_rows;
  bst_uint         n_features;

  // Get SHAP values for one sample (sorted by |shap| desc)
  std::vector<std::pair<std::string, double>> top_features(bst_uint row,
                      bst_uint k = 0) const;
};

// =============================================================================
//  Metrics — output of XGBModel::score()
// =============================================================================
struct Metrics {
  std::string task;
  double rmse  {0};
  double mae   {0};
  double r2  {0};
  double auc   {0};
  double accuracy{0};
  double logloss {0};
  double train_ms{0};
  double infer_ms{0};

  void print() const;
};

// =============================================================================
//  XGBModel — high-level sklearn-style API
//
//  All setters return *this so they can be chained:
//  model.n_estimators(200).learning_rate(0.05f).max_depth(6);
// =============================================================================
class XGBModel {
public:
  // Construction          
  explicit XGBModel(const std::string& task = "regression");

  // Hyperparameter setters (fluent interface)      
  XGBModel& task     (const std::string& t);
  XGBModel& n_estimators (int n);
  XGBModel& learning_rate  (float lr);
  XGBModel& max_depth  (int d);
  XGBModel& subsample  (float s);
  XGBModel& colsample  (float c); // colsample_bytree
  XGBModel& lambda   (float l); // L2 regularisation
  XGBModel& gamma    (float g); // min split gain
  XGBModel& grow_policy  (const std::string& p);  // "depthwise" | "lossguide"
  XGBModel& num_class  (int n); // for multi:softmax only
  XGBModel& feature_names  (const std::vector<std::string>& names);

  // Training behaviour         
  XGBModel& early_stopping (int patience, bool higher_is_better = false);
  XGBModel& checkpoint   (const std::string& path);  // save best model
  XGBModel& verbose    (int every = 1);    // 0 = silent
  XGBModel& lr_cosine  (float eta_max, float eta_min, int t_max);
  XGBModel& lr_step    (float initial,  float drop, int step_size);

  // Fit            
  // From raw matrices
  XGBModel& fit(const std::vector<std::vector<bst_float>>& X,
      const std::vector<bst_float>&     y);
  XGBModel& fit(const core::MatrixView& X,
      const std::vector<bst_float>& y);

  XGBModel& fit(const std::vector<std::vector<bst_float>>& X_train,
      const std::vector<bst_float>&     y_train,
      const std::vector<std::vector<bst_float>>& X_val,
      const std::vector<bst_float>&     y_val);
  XGBModel& fit(const core::MatrixView& X_train,
      const std::vector<bst_float>& y_train,
      const core::MatrixView& X_val,
      const std::vector<bst_float>& y_val);

  // From DMatrix
  XGBModel& fit(std::shared_ptr<DMatrix> train_dm,
      std::shared_ptr<DMatrix> val_dm = nullptr);

  // From CSV file  (label_col: -1 = last column)
  XGBModel& fit_csv(const std::string& train_path,
        const std::string& val_path = "",
        int      label_col  = -1);

  // Predict          
  std::vector<bst_float> predict(const std::vector<std::vector<bst_float>>& X) const;
  std::vector<bst_float> predict(const core::MatrixView& X) const;
  std::vector<bst_float> predict(std::shared_ptr<DMatrix> dm) const;
  std::vector<bst_float> predict_csv(const std::string& path, int label_col = -1) const;

  // Score            
  Metrics score(const std::vector<std::vector<bst_float>>& X,
      const std::vector<bst_float>&     y) const;
  Metrics score(const core::MatrixView& X,
      const std::vector<bst_float>& y) const;
  Metrics score(std::shared_ptr<DMatrix> dm) const;

  // Feature Importance         
  // Returns vector<{name, value}> sorted by importance desc
  // type: "gain" | "weight" | "cover"
  std::vector<std::pair<std::string, bst_float>>
    feature_importances(const std::string& type = "gain") const;

  // SHAP            
  ShapResult explain(const std::vector<std::vector<bst_float>>& X,
         bst_uint max_rows = 300) const;
  ShapResult explain(const core::MatrixView& X,
         bst_uint max_rows = 300) const;
  ShapResult explain(std::shared_ptr<DMatrix> dm,
         bst_uint max_rows = 300) const;

  // Persistence           
  void save   (const std::string& path) const;  // auto: .json or .bin
  void load   (const std::string& path);
  void save_binary(const std::string& path) const;
  void load_binary(const std::string& path);

  // Dashboard export          
  // Writes all CSV/JSON dashboard files to output_dir/
  // Requires that fit() and predict(test_dm) have been called
  void export_dashboard(const std::string& output_dir,
          std::shared_ptr<DMatrix> test_dm = nullptr) const;

  // Accessors          
  bool         is_fitted()  const { return booster_ != nullptr; }
  bst_uint       num_trees()  const;
  bst_uint       num_features() const;
  const TrainHistory&    history()  const;
  std::shared_ptr<GradientBooster> booster() const  { return booster_; }
  const std::string&     task_name() const  { return task_; }

  // Pretty-print training summary
  void print_summary() const;

private:
  // Config state           
  std::string task_     {"regression"};
  int     n_estimators_ {200};
  float   learning_rate_  {0.1f};
  int     max_depth_  {5};
  float   subsample_  {1.0f};
  float   colsample_  {1.0f};
  float   lambda_   {50.0f};
  float   gamma_    {0.0f};
  std::string grow_policy_  {"depthwise"};
  int     num_class_  {2};
  std::vector<std::string> feature_names_;

  // Callbacks config
  int   early_stopping_rounds_ {0};    // 0 = disabled
  bool    es_higher_is_better_ {false};
  std::string checkpoint_path_;
  int   verbose_every_   {25};
  bool    use_lr_cosine_   {false};
  float   lr_cos_max_    {0.1f};
  float   lr_cos_min_    {0.01f};
  int   lr_cos_tmax_     {100};
  bool    use_lr_step_     {false};
  float   lr_step_initial_   {0.1f};
  float   lr_step_drop_    {0.5f};
  int   lr_step_size_    {50};

  // Runtime state          
  std::shared_ptr<GradientBooster>  booster_;
  std::shared_ptr<DMatrix>    last_train_dm_;
  std::shared_ptr<DMatrix>    last_val_dm_;
  std::shared_ptr<TrainHistory>   history_;
  double          train_ms_  {0};
  double          infer_ms_  {0};

  // Internal helpers         
  Config   build_config()  const;
  std::string metric_name() const;
  bool    higher_is_better() const;

  Metrics compute_metrics(const std::vector<bst_float>& y_true,
           const std::vector<bst_float>& y_pred) const;

  void write_dashboard_files(const std::string& dir,
            std::shared_ptr<DMatrix> test_dm,
            const std::vector<bst_float>& preds) const;

  static std::string jstr(const std::string& s);
  static std::string jkvs(const std::string& k, const std::string& v);
  static std::string jkvn(const std::string& k, double v);
  static std::string jkv (const std::string& k, const std::string& v);
  static std::string jarr(const std::vector<std::string>& v);
  static std::string op  (const std::string& dir, const std::string& f);
};

// =============================================================================
//  Convenience aliases
// =============================================================================
class XGBRegressor : public XGBModel {
public:
  XGBRegressor() : XGBModel("regression") {}
};

class XGBClassifier : public XGBModel {
public:
  // binary classification by default; pass num_class > 2 for multiclass
  explicit XGBClassifier(int num_class = 2)
    : XGBModel(num_class > 2 ? "multiclass" : "binary")
  {
    if (num_class > 2) this->num_class(num_class);
  }
};

class XGBTweedieRegressor : public XGBModel {
public:
  XGBTweedieRegressor() : XGBModel("tweedie") {}
};

} // namespace xgb
