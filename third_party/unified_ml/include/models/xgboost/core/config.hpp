#pragma once
#include "models/xgboost/core/types.hpp"
#include <string>
#include <unordered_map>

namespace xgb {

//                        
//  TreeConfig
//  Per-tree hyperparameters.
//  Maps directly to XGBoost's tree_method params.
//                        
struct TreeConfig {
    bst_int   max_depth        {6};       // maximum tree depth
    bst_float min_child_weight {1.f};     // minimum sum of hessian in a leaf
    bst_float gamma            {0.f};     // minimum loss reduction (γ, Eq. 7)
    bst_float lambda           {1.f};     // L2 regularisation on leaf weights
    bst_float alpha            {0.f};     // L1 regularisation (future)
    bst_float colsample_bytree {1.f};     // fraction of features per tree
    bst_float colsample_bylevel{1.f};     // fraction of features per level
    bst_float subsample        {1.f};     // fraction of rows per tree

    // Approximate split finding (Sec. 3.2)
    bool      use_approx       {false};
    bst_float sketch_eps       {0.03f};   // ε in weighted quantile sketch

    // Sparsity-aware (Sec. 3.4)
    bool      sparsity_aware   {true};

    //   Growth policy                            
    // "depthwise"  — level-by-level growth (default, XGBoost-style)
    // "lossguide"  — best-first leaf expansion (LightGBM-style, Feature 9)
    std::string grow_policy    {"depthwise"};
    bst_int   max_leaves       {0};    // 0 = unlimited (only used for lossguide)

    //   Threading / sampling (propagated from BoosterConfig)        
    // These are copied into TreeConfig by GradientBooster::boost_one_round()
    // so that DecisionTree / SplitEvaluator can access them without needing
    // a reference to the full BoosterConfig.
    bst_uint  seed             {0};      // RNG seed for colsample_bylevel
    bst_int   nthread          {-1};     // OpenMP thread count; -1 = all cores
};

//                        
//  BoosterConfig
//  Ensemble / training hyperparameters.
//                        
struct BoosterConfig {
    bst_int   num_round        {100};     // number of boosting iterations K
    bst_float eta              {0.3f};    // shrinkage / learning rate η (Sec. 2.3)
    bst_float base_score       {0.5f};    // global bias ŷ(0)
    bst_uint  seed             {0};
    bst_int   nthread          {-1};      // -1 = use all cores (future: OMP)
    std::string objective      {"reg:squarederror"};
    std::string eval_metric    {"rmse"};
    std::string tree_method    {"exact"}; // exact | approx | hist (future)
    TaskType  task             {TaskType::kRegression};

    //   Extended params (new features)                   
    bst_int   num_class        {2};       // for multi:softmax (Feature 1)
    bst_float tweedie_variance_power {1.5f}; // for reg:tweedie (Feature 8)

    TreeConfig tree;
};

//                        
//  Config — unified config object
//  Wraps BoosterConfig; can be loaded from file.
//                        
class Config {
public:
    Config() = default;
    explicit Config(const std::string& config_path);

    // Fluent setters              
    Config& set(const std::string& key, const std::string& value);
    Config& set_objective(const std::string& obj);
    Config& set_num_round(bst_int n);
    Config& set_eta(bst_float eta);
    Config& set_max_depth(bst_int d);
    Config& set_lambda(bst_float lambda);
    Config& set_gamma(bst_float gamma);
    Config& set_subsample(bst_float s);
    Config& set_colsample(bst_float c);
    Config& set_min_child_weight(bst_float w);

    // Accessors                 
    const BoosterConfig& booster()  const { return booster_; }
    const TreeConfig&    tree()     const { return booster_.tree; }
    BoosterConfig&       booster()        { return booster_; }
    TreeConfig&          tree()           { return booster_.tree; }

    void print() const;

private:
    BoosterConfig booster_;
    std::unordered_map<std::string, std::string> raw_params_;

    void apply_raw_params();
};

} // namespace xgb
