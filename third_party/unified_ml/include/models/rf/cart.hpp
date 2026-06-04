#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include "models/rf/dataset.hpp"
#include "models/rf/tree.hpp"
#include "models/rf/split.hpp"
#include "models/rf/utils.hpp"

namespace rf {

// ---------------------------------------------------------------------------
// Hyperparameters for a single CART tree
// ---------------------------------------------------------------------------
struct CartParams {
    int    max_depth{-1};           // -1 = unlimited
    int    min_samples_split{2};    // Minimum samples required to split a node
    int    min_samples_leaf{1};     // Minimum samples in each leaf after split
    double min_impurity_decrease{0.0}; // Minimum gain to consider a split
    size_t max_features{0};         // 0 = use all features; >0 = random subset size
    uint64_t random_seed{42};

    // Feature subset strategy for Random Forest
    enum class MaxFeaturesStrategy { All, Sqrt, Log2, Custom };
    MaxFeaturesStrategy max_features_strategy{MaxFeaturesStrategy::All};

    size_t resolve_max_features(size_t n_features) const;
};

// ---------------------------------------------------------------------------
// CART Decision Tree (Classification and Regression)
// ---------------------------------------------------------------------------
class CartTree {
public:
    explicit CartTree(CartParams params = {});

    // Fit the tree on a full Dataset
    void fit(const Dataset& dataset);

    // Fit using only a subset of row indices (used by Random Forest)
    void fit(const Dataset& dataset, const std::vector<size_t>& indices);

    // Predict for a single sample
    double predict_one(const std::vector<double>& x) const;

    // Predict for all rows in a Dataset
    std::vector<double> predict(const Dataset& dataset) const;

    // Predict for a matrix X
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    std::vector<double> predict(const core::MatrixView& X) const;

    // Feature importance (unnormalized raw scores from this tree)
    std::vector<double> feature_importances(size_t n_features) const;

    const TreeNode* root() const { return root_.get(); }
    const CartParams& params() const { return params_; }
    void set_serialized_state(std::unique_ptr<TreeNode> root, TaskType task, int n_classes, size_t n_features);

    int depth() const;

private:
    CartParams params_;
    std::unique_ptr<TreeNode> root_;
    TaskType task_{TaskType::Classification};
    int n_classes_{0};
    size_t n_features_{0};
    RNG rng_;

    std::unique_ptr<TreeNode> build_node(const Dataset& dataset,
                                          const std::vector<size_t>& indices,
                                          int current_depth);

    double leaf_prediction(const Dataset& dataset,
                           const std::vector<size_t>& indices) const;

    // Compute depth recursively
    static int node_depth(const TreeNode* node);
};

} // namespace rf
