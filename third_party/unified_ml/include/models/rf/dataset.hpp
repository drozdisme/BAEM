#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include "models/rf/utils.hpp"
#include "core/matrix_view.hpp"

namespace rf {

// ---------------------------------------------------------------------------
// Task type
// ---------------------------------------------------------------------------
enum class TaskType { Classification, Regression };

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------
class Dataset {
public:
    // Construct from feature matrix and label vector
    Dataset(std::vector<std::vector<double>> X,
            std::vector<double> y,
            TaskType task = TaskType::Classification);
    Dataset(core::MatrixView X,
            std::vector<double> y,
            TaskType task = TaskType::Classification);

    // Getters
    size_t n_samples()  const { return X_.size(); }
    size_t n_features() const { return n_features_; }
    int    n_classes()  const { return n_classes_; }  // Meaningful only for classification
    TaskType task()     const { return task_; }

    const std::vector<std::vector<double>>& X() const { return X_; }
    const std::vector<double>&              y() const { return y_; }
    core::MatrixView                        X_view() const;

    // Access single sample features
    const std::vector<double>& row(size_t i) const { return X_[i]; }
    double label(size_t i)                   const { return y_[i]; }
    double feature(size_t sample, size_t feat) const { return X_flat_[sample * n_features_ + feat]; }

    // ---------------------------------------------------------------------------
    // Splits
    // ---------------------------------------------------------------------------

    // Random train/test split — returns (train_indices, test_indices)
    std::pair<std::vector<size_t>, std::vector<size_t>>
    train_test_split(double test_ratio, RNG& rng) const;

    // Bootstrap sample (with replacement) — returns sampled indices
    // Also populates out_of_bag with indices NOT selected
    std::vector<size_t> bootstrap_sample(RNG& rng,
                                          std::vector<size_t>* out_of_bag = nullptr) const;

    // Sub-dataset view from index list (copies data)
    Dataset subset(const std::vector<size_t>& indices) const;

    // ---------------------------------------------------------------------------
    // Column statistics (useful for split enumeration)
    // ---------------------------------------------------------------------------
    std::vector<double> unique_values(size_t feature_idx,
                                       const std::vector<size_t>& indices) const;

private:
    std::vector<std::vector<double>> X_;
    std::vector<double>              X_flat_;
    std::vector<double>              y_;
    size_t   n_features_{0};
    int      n_classes_{0};
    TaskType task_;

    void compute_n_classes();
};

} // namespace rf
