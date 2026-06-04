#include "models/xgboost/core/config.hpp"
#include "models/xgboost/utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace xgb {

Config::Config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config: " + config_path);

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // Trim whitespace
        auto trim = [](std::string& s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
            while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t')) s.pop_back();
        };
        trim(key); trim(val);
        raw_params_[key] = val;
    }
    apply_raw_params();
}

Config& Config::set(const std::string& key, const std::string& value) {
    raw_params_[key] = value;
    apply_raw_params();
    return *this;
}

Config& Config::set_objective(const std::string& obj) {
    booster_.objective = obj;
    if (obj == "binary:logistic")
        booster_.task = TaskType::kBinaryClassification;
    else
        booster_.task = TaskType::kRegression;
    return *this;
}

Config& Config::set_num_round(bst_int n)         { booster_.num_round = n;            return *this; }
Config& Config::set_eta(bst_float eta)            { booster_.eta = eta;               return *this; }
Config& Config::set_max_depth(bst_int d)          { booster_.tree.max_depth = d;      return *this; }
Config& Config::set_lambda(bst_float lambda)      { booster_.tree.lambda = lambda;    return *this; }
Config& Config::set_gamma(bst_float gamma)        { booster_.tree.gamma = gamma;      return *this; }
Config& Config::set_subsample(bst_float s)        { booster_.tree.subsample = s;      return *this; }
Config& Config::set_colsample(bst_float c)        { booster_.tree.colsample_bytree = c; return *this; }
Config& Config::set_min_child_weight(bst_float w) { booster_.tree.min_child_weight = w; return *this; }

void Config::apply_raw_params() {
    for (const auto& [k, v] : raw_params_) {
        if      (k == "num_round"          || k == "n_estimators") booster_.num_round          = std::stoi(v);
        else if (k == "eta"                || k == "learning_rate") booster_.eta               = std::stof(v);
        else if (k == "max_depth")                                   booster_.tree.max_depth    = std::stoi(v);
        else if (k == "lambda"             || k == "reg_lambda")     booster_.tree.lambda       = std::stof(v);
        else if (k == "alpha"              || k == "reg_alpha")      booster_.tree.alpha        = std::stof(v);
        else if (k == "gamma"              || k == "min_split_loss") booster_.tree.gamma        = std::stof(v);
        else if (k == "min_child_weight")                            booster_.tree.min_child_weight = std::stof(v);
        else if (k == "subsample")                                   booster_.tree.subsample    = std::stof(v);
        else if (k == "colsample_bytree")                            booster_.tree.colsample_bytree = std::stof(v);
        else if (k == "base_score")                                  booster_.base_score        = std::stof(v);
        else if (k == "seed"               || k == "random_state")   booster_.seed              = static_cast<bst_uint>(std::stoul(v));
        else if (k == "objective")         set_objective(v);
        else if (k == "eval_metric")                                 booster_.eval_metric       = v;
        else if (k == "tree_method")                                 booster_.tree_method       = v;
        else if (k == "nthread")                                     booster_.nthread           = std::stoi(v);
        //   New params                           
        else if (k == "grow_policy")                                 booster_.tree.grow_policy  = v;
        else if (k == "max_leaves")                                  booster_.tree.max_leaves   = std::stoi(v);
        else if (k == "tweedie_variance_power")                      booster_.tweedie_variance_power = std::stof(v);
        else if (k == "num_class")                                   booster_.num_class         = std::stoi(v);
    }
}

void Config::print() const {
    std::cout << "  XGBoost Config              \n";
    std::cout << "  num_round        : " << booster_.num_round          << "\n";
    std::cout << "  eta              : " << booster_.eta                 << "\n";
    std::cout << "  objective        : " << booster_.objective           << "\n";
    std::cout << "  eval_metric      : " << booster_.eval_metric         << "\n";
    std::cout << "  max_depth        : " << booster_.tree.max_depth      << "\n";
    std::cout << "  min_child_weight : " << booster_.tree.min_child_weight << "\n";
    std::cout << "  gamma            : " << booster_.tree.gamma          << "\n";
    std::cout << "  lambda           : " << booster_.tree.lambda         << "\n";
    std::cout << "  subsample        : " << booster_.tree.subsample      << "\n";
    std::cout << "  colsample_bytree : " << booster_.tree.colsample_bytree << "\n";
    std::cout << "                      \n";
}

} // namespace xgb
