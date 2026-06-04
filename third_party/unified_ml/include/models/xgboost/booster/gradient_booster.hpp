#pragma once
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/core/config.hpp"
#include "models/xgboost/tree/decision_tree.hpp"
#include "models/xgboost/tree/tree_node.hpp"
#include "models/xgboost/objective/loss_function.hpp"
#include "models/xgboost/utils/xoshiro256.hpp"
#include <vector>
#include <memory>
#include <string>

namespace xgb {

class DMatrix;

class GradientBooster {
public:
    explicit GradientBooster(const BoosterConfig& cfg);

    bst_float boost_one_round(const DMatrix& dm, std::vector<bst_float>& scores);

    bst_float predict_raw(const DMatrix& dm, bst_uint row) const;
    std::vector<bst_float> predict_batch_raw(const DMatrix& dm, const std::vector<bst_uint>& row_indices) const;

    bst_uint num_trees() const { return static_cast<bst_uint>(trees_.size()); }

    // Trim the tree ensemble to the first n trees (for best-iteration rollback)
    void trim_trees(bst_uint n) {
        if (n < static_cast<bst_uint>(trees_.size()))
            trees_.resize(n);
    }
    const std::vector<std::unique_ptr<DecisionTree>>& trees() const { return trees_; }

    std::vector<bst_float> feature_importance(bst_uint n_features) const;

    void save_model(const std::string& path) const;
    void load_model(const std::string& path);
    void save_model_binary(const std::string& path) const;
    void load_model_binary(const std::string& path);

    // For deserialization: inject a pre-built node list as a new tree
    void inject_tree(std::vector<TreeNode> nodes);
    void clear_trees() { trees_.clear(); }

    std::string dump_model_text() const;
    std::string dump_model_json() const;

    bst_float base_score() const { return base_score_; }
    void set_base_score(bst_float s) { base_score_ = s; }
    const BoosterConfig& config() const { return cfg_; }
    BoosterConfig& mutable_config() { return cfg_; }

private:
    BoosterConfig cfg_;
    std::vector<std::unique_ptr<DecisionTree>> trees_;
    bst_float base_score_{0.5f};
    std::unique_ptr<LossFunction> loss_fn_;
    mutable Xoshiro256 rng_;  // P4-6: xoshiro256** replaces LCG

    std::vector<bst_uint> sample_columns(bst_uint n_total) const;
    std::vector<bst_uint> sample_rows(bst_uint n_total) const;
};

} // namespace xgb
