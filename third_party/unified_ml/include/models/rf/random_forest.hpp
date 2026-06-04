#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include "models/rf/cart.hpp"
#include "models/rf/dataset.hpp"
#include "models/rf/utils.hpp"

namespace rf {

// ---------------------------------------------------------------------------
// Hyperparameters for Random Forest
// ---------------------------------------------------------------------------
struct RandomForestParams {
    int    n_estimators{100};
    int    max_depth{-1};
    int    min_samples_split{2};
    int    min_samples_leaf{1};
    double min_impurity_decrease{0.0};
    CartParams::MaxFeaturesStrategy max_features_strategy{CartParams::MaxFeaturesStrategy::Sqrt};
    size_t max_features{0};           // Used if strategy == Custom
    bool   bootstrap{true};           // Use bootstrap sampling
    bool   compute_oob{true};         // Compute OOB error
    uint64_t random_seed{42};

    // Verbose progress callback (optional)
    // Called with (tree_index, n_estimators)
    std::function<void(int, int)> progress_callback;

    void validate() const;
};

// ---------------------------------------------------------------------------
// Random Forest (Classification and Regression)
// ---------------------------------------------------------------------------
class RandomForest {
public:
    explicit RandomForest(RandomForestParams params = {});

    void fit(const Dataset& dataset);

    // Predict class labels (classification) or values (regression)
    std::vector<double> predict(const Dataset& dataset) const;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    std::vector<double> predict(const core::MatrixView& X) const;
    double              predict_one(const std::vector<double>& x) const;

    // Predict class probabilities (classification only)
    // Returns (n_samples × n_classes) matrix
    std::vector<std::vector<double>> predict_proba(const Dataset& dataset) const;
    std::vector<double>              predict_proba_flat(const Dataset& dataset) const;

    // OOB error (computed during fit if compute_oob=true and bootstrap=true)
    double oob_error() const;
    bool   has_oob()   const { return oob_computed_; }

    // Aggregated feature importance across all trees
    std::vector<double> feature_importances() const;

    // Accessors
    size_t n_trees()    const { return trees_.size(); }
    size_t n_features() const { return n_features_; }
    int    n_classes()  const { return n_classes_; }
    TaskType task()     const { return task_; }

    const CartTree& tree(size_t i) const { return *trees_[i]; }

    // Versioned binary serialization
    void save(const std::string& filepath) const;
    static RandomForest load(const std::string& filepath);

private:
    RandomForestParams params_;
    std::vector<std::unique_ptr<CartTree>> trees_;

    TaskType task_{TaskType::Classification};
    int      n_classes_{0};
    size_t   n_features_{0};

    // OOB prediction accumulator
    // oob_sums_[i]  = sum of predictions for sample i from OOB trees
    // oob_counts_[i] = number of OOB trees that covered sample i
    std::vector<double> oob_sums_;
    std::vector<int>    oob_counts_;
    // For classification, store vote counts
    std::vector<std::vector<int>> oob_votes_;
    double oob_error_{0.0};
    bool   oob_computed_{false};

    CartParams make_tree_params(uint64_t seed) const;
    void       compute_oob_error(const Dataset& dataset);
};

} // namespace rf
