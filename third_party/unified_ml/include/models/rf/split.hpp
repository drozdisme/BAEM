#pragma once

#include <vector>
#include <limits>
#include <cstddef>
#include "models/rf/dataset.hpp"
#include "models/rf/utils.hpp"

namespace rf {

// ---------------------------------------------------------------------------
// Result of finding the best split
// ---------------------------------------------------------------------------
struct SplitResult {
    int    feature_index{-1};
    double threshold{0.0};
    double gain{-std::numeric_limits<double>::infinity()};  // Impurity reduction
    double impurity_left{0.0};
    double impurity_right{0.0};
    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;

    bool valid() const { return feature_index >= 0; }
};

// ---------------------------------------------------------------------------
// Impurity functions (stateless, free functions)
// ---------------------------------------------------------------------------

// Gini impurity for a set of labels (classification)
double gini_impurity(const std::vector<double>& y,
                     const std::vector<size_t>& indices,
                     int n_classes);

// Mean Squared Error for a set of labels (regression)
double mse_impurity(const std::vector<double>& y,
                    const std::vector<size_t>& indices);

// Entropy (for reference; not used by default but available)
double entropy_impurity(const std::vector<double>& y,
                        const std::vector<size_t>& indices,
                        int n_classes);

// ---------------------------------------------------------------------------
// Split finder
// ---------------------------------------------------------------------------
class SplitFinder {
public:
    struct Params {
        int    min_samples_leaf{1};
        size_t max_thresholds{0};  // 0 = try all unique midpoints
    };

    SplitFinder() = default;
    explicit SplitFinder(Params p) : params_(p) {}

    // Find the best split across all candidate features for classification
    SplitResult best_split_classification(const Dataset& dataset,
                                           const std::vector<size_t>& indices,
                                           const std::vector<size_t>& feature_subset,
                                           int n_classes) const;

    // Find the best split across all candidate features for regression
    SplitResult best_split_regression(const Dataset& dataset,
                                       const std::vector<size_t>& indices,
                                       const std::vector<size_t>& feature_subset) const;

private:
    Params params_;

    // Try all thresholds for a single feature (classification)
    SplitResult best_threshold_classification(const Dataset& dataset,
                                               const std::vector<size_t>& indices,
                                               size_t feature_idx,
                                               int n_classes) const;

    // Try all thresholds for a single feature (regression)
    SplitResult best_threshold_regression(const Dataset& dataset,
                                           const std::vector<size_t>& indices,
                                           size_t feature_idx) const;
};

} // namespace rf
