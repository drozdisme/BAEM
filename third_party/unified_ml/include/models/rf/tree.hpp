#pragma once

#include <memory>
#include <vector>
#include <cstddef>

namespace rf {

// ---------------------------------------------------------------------------
// A single node in a decision tree
// ---------------------------------------------------------------------------
struct TreeNode {
    // --- Split information (internal node) ---
    int    feature_index{-1};   // Which feature to split on
    double threshold{0.0};      // Split threshold: go left if x[feature_index] <= threshold

    // --- Leaf prediction ---
    bool   is_leaf{false};
    double prediction{0.0};     // For regression: mean value; for classification: majority class

    // --- Children ---
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    // --- Meta (optional, for feature importance) ---
    double impurity_decrease{0.0};  // Weighted impurity reduction at this split
    size_t n_samples{0};            // Samples passing through this node

    TreeNode() = default;

    // Non-copyable (unique_ptr children)
    TreeNode(const TreeNode&) = delete;
    TreeNode& operator=(const TreeNode&) = delete;

    TreeNode(TreeNode&&) = default;
    TreeNode& operator=(TreeNode&&) = default;

    bool is_internal() const { return !is_leaf; }
};

// ---------------------------------------------------------------------------
// Feature importance accumulator
// ---------------------------------------------------------------------------
struct FeatureImportance {
    std::vector<double> scores;   // Raw importance per feature (sum of impurity_decrease * n_samples)
    size_t n_features{0};

    explicit FeatureImportance(size_t n) : scores(n, 0.0), n_features(n) {}

    void accumulate(const TreeNode* node);
    void normalize();
    std::vector<double> get() const { return scores; }
};

} // namespace rf
