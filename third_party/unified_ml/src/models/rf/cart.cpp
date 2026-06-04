#include "models/rf/cart.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace rf {

// ---------------------------------------------------------------------------
// CartParams helpers
// ---------------------------------------------------------------------------
size_t CartParams::resolve_max_features(size_t n_features) const {
    switch (max_features_strategy) {
        case MaxFeaturesStrategy::All:
            return n_features;
        case MaxFeaturesStrategy::Sqrt:
            return std::max(size_t(1), static_cast<size_t>(std::sqrt(static_cast<double>(n_features))));
        case MaxFeaturesStrategy::Log2:
            return std::max(size_t(1), static_cast<size_t>(std::log2(static_cast<double>(n_features))));
        case MaxFeaturesStrategy::Custom:
            return (max_features > 0 && max_features <= n_features) ? max_features : n_features;
    }
    return n_features;
}

// ---------------------------------------------------------------------------
// CartTree
// ---------------------------------------------------------------------------
CartTree::CartTree(CartParams params)
    : params_(std::move(params)), rng_(params_.random_seed) {}

void CartTree::fit(const Dataset& dataset) {
    std::vector<size_t> all = iota_indices(dataset.n_samples());
    fit(dataset, all);
}

void CartTree::fit(const Dataset& dataset, const std::vector<size_t>& indices) {
    if (indices.empty()) throw std::invalid_argument("CartTree::fit: empty indices");

    task_       = dataset.task();
    n_classes_  = dataset.n_classes();
    n_features_ = dataset.n_features();

    root_ = build_node(dataset, indices, 0);
}

// ---------------------------------------------------------------------------
// Recursive tree builder
// ---------------------------------------------------------------------------
std::unique_ptr<TreeNode> CartTree::build_node(const Dataset& dataset,
                                                 const std::vector<size_t>& indices,
                                                 int current_depth) {
    auto node = std::make_unique<TreeNode>();
    node->n_samples = indices.size();

    // --- Stopping conditions ---
    bool stop = false;

    // Max depth reached
    if (params_.max_depth >= 0 && current_depth >= params_.max_depth)
        stop = true;

    // Too few samples to split
    if (!stop && static_cast<int>(indices.size()) < params_.min_samples_split)
        stop = true;

    // All labels identical (pure node)
    if (!stop) {
        bool all_same = true;
        double first = dataset.y()[indices[0]];
        for (size_t i : indices) {
            if (dataset.y()[i] != first) { all_same = false; break; }
        }
        if (all_same) stop = true;
    }

    if (stop) {
        node->is_leaf    = true;
        node->prediction = leaf_prediction(dataset, indices);
        return node;
    }

    // --- Feature subsampling ---
    size_t k = params_.resolve_max_features(n_features_);
    std::vector<size_t> feature_subset;
    if (k >= n_features_) {
        feature_subset = iota_indices(n_features_);
    } else {
        feature_subset = sample_without_replacement(n_features_, k, rng_);
    }

    // --- Find best split ---
    SplitFinder::Params sp;
    sp.min_samples_leaf = params_.min_samples_leaf;
    SplitFinder finder(sp);

    SplitResult split;
    if (task_ == TaskType::Classification) {
        split = finder.best_split_classification(dataset, indices, feature_subset, n_classes_);
    } else {
        split = finder.best_split_regression(dataset, indices, feature_subset);
    }

    // If no valid split found, or gain below threshold → leaf
    if (!split.valid() || split.gain < params_.min_impurity_decrease) {
        node->is_leaf    = true;
        node->prediction = leaf_prediction(dataset, indices);
        return node;
    }

    // --- Populate internal node ---
    node->feature_index     = split.feature_index;
    node->threshold         = split.threshold;
    node->impurity_decrease = split.gain;

    // --- Recurse ---
    node->left  = build_node(dataset, split.left_indices,  current_depth + 1);
    node->right = build_node(dataset, split.right_indices, current_depth + 1);

    return node;
}

// ---------------------------------------------------------------------------
// Leaf prediction
// ---------------------------------------------------------------------------
double CartTree::leaf_prediction(const Dataset& dataset,
                                  const std::vector<size_t>& indices) const {
    if (task_ == TaskType::Regression) {
        return mean(dataset.y(), indices);
    } else {
        return static_cast<double>(majority_class(dataset.y(), indices));
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
double CartTree::predict_one(const std::vector<double>& x) const {
    if (!root_) throw std::runtime_error("CartTree not fitted");
    const TreeNode* node = root_.get();
    while (!node->is_leaf) {
        if (x[static_cast<size_t>(node->feature_index)] <= node->threshold)
            node = node->left.get();
        else
            node = node->right.get();
    }
    return node->prediction;
}

std::vector<double> CartTree::predict(const Dataset& dataset) const {
    std::vector<double> preds(dataset.n_samples());
    for (size_t i = 0; i < dataset.n_samples(); ++i)
        preds[i] = predict_one(dataset.row(i));
    return preds;
}

std::vector<double> CartTree::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> preds(X.size());
    for (size_t i = 0; i < X.size(); ++i)
        preds[i] = predict_one(X[i]);
    return preds;
}

std::vector<double> CartTree::predict(const core::MatrixView& X) const {
    std::vector<double> preds(X.rows);
    std::vector<double> row(X.cols, 0.0);
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) row[j] = X(i, j);
        preds[i] = predict_one(row);
    }
    return preds;
}

// ---------------------------------------------------------------------------
// Feature importance
// ---------------------------------------------------------------------------
std::vector<double> CartTree::feature_importances(size_t n_features) const {
    FeatureImportance fi(n_features);
    fi.accumulate(root_.get());
    fi.normalize();
    return fi.get();
}

// ---------------------------------------------------------------------------
// Depth
// ---------------------------------------------------------------------------
int CartTree::node_depth(const TreeNode* node) {
    if (!node || node->is_leaf) return 0;
    return 1 + std::max(node_depth(node->left.get()), node_depth(node->right.get()));
}

int CartTree::depth() const {
    return node_depth(root_.get());
}

void CartTree::set_serialized_state(std::unique_ptr<TreeNode> root,
                                    TaskType task,
                                    int n_classes,
                                    size_t n_features) {
    root_ = std::move(root);
    task_ = task;
    n_classes_ = n_classes;
    n_features_ = n_features;
}

} // namespace rf
