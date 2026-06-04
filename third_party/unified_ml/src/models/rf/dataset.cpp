#include "models/rf/dataset.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace rf {

Dataset::Dataset(std::vector<std::vector<double>> X,
                 std::vector<double> y,
                 TaskType task)
    : X_(std::move(X)), y_(std::move(y)), task_(task)
{
    if (X_.empty()) throw std::invalid_argument("Dataset: empty X");
    if (X_.size() != y_.size()) throw std::invalid_argument("Dataset: X and y size mismatch");

    n_features_ = X_[0].size();
    for (size_t i = 1; i < X_.size(); ++i) {
        if (X_[i].size() != n_features_)
            throw std::invalid_argument("Dataset: inconsistent feature count in row " + std::to_string(i));
    }
    X_flat_.resize(X_.size() * n_features_);
    for (size_t i = 0; i < X_.size(); ++i)
        std::copy(X_[i].begin(), X_[i].end(), X_flat_.begin() + i * n_features_);

    compute_n_classes();
}

Dataset::Dataset(core::MatrixView X,
                 std::vector<double> y,
                 TaskType task)
    : y_(std::move(y)), n_features_(X.cols), task_(task)
{
    if (!X.data || X.rows == 0) throw std::invalid_argument("Dataset: empty X");
    if (X.cols == 0) throw std::invalid_argument("Dataset: X has zero features");
    if (X.rows != y_.size()) throw std::invalid_argument("Dataset: X and y size mismatch");

    X_.assign(X.rows, std::vector<double>(X.cols, 0.0));
    X_flat_.resize(X.rows * X.cols);
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) {
            const double v = X(i, j);
            X_[i][j] = v;
            X_flat_[i * X.cols + j] = v;
        }
    }
    compute_n_classes();
}

core::MatrixView Dataset::X_view() const {
    return core::MatrixView(X_flat_.data(), n_samples(), n_features_, n_features_);
}

void Dataset::compute_n_classes() {
    if (task_ == TaskType::Regression) {
        n_classes_ = 0;
        return;
    }
    int max_cls = 0;
    for (double v : y_) {
        int c = static_cast<int>(v);
        if (c < 0) throw std::invalid_argument("Dataset: negative class label");
        max_cls = std::max(max_cls, c);
    }
    n_classes_ = max_cls + 1;
}

std::pair<std::vector<size_t>, std::vector<size_t>>
Dataset::train_test_split(double test_ratio, RNG& rng) const {
    if (test_ratio <= 0.0 || test_ratio >= 1.0)
        throw std::invalid_argument("test_ratio must be in (0, 1)");

    std::vector<size_t> indices = iota_indices(n_samples());
    rng.shuffle(indices);

    size_t n_test  = static_cast<size_t>(std::round(test_ratio * static_cast<double>(n_samples())));
    size_t n_train = n_samples() - n_test;

    std::vector<size_t> train(indices.begin(), indices.begin() + n_train);
    std::vector<size_t> test (indices.begin() + n_train, indices.end());
    return {std::move(train), std::move(test)};
}

std::vector<size_t> Dataset::bootstrap_sample(RNG& rng, std::vector<size_t>* out_of_bag) const {
    size_t n = n_samples();
    std::vector<size_t> sampled = bootstrap_indices(n, rng);

    if (out_of_bag) {
        std::unordered_set<size_t> sampled_set(sampled.begin(), sampled.end());
        out_of_bag->clear();
        for (size_t i = 0; i < n; ++i) {
            if (sampled_set.find(i) == sampled_set.end())
                out_of_bag->push_back(i);
        }
    }
    return sampled;
}

Dataset Dataset::subset(const std::vector<size_t>& indices) const {
    std::vector<std::vector<double>> newX;
    std::vector<double>              newY;
    newX.reserve(indices.size());
    newY.reserve(indices.size());
    for (size_t i : indices) {
        newX.push_back(X_[i]);
        newY.push_back(y_[i]);
    }
    return Dataset(std::move(newX), std::move(newY), task_);
}

std::vector<double> Dataset::unique_values(size_t feature_idx,
                                            const std::vector<size_t>& indices) const {
    std::vector<double> vals;
    vals.reserve(indices.size());
    for (size_t i : indices) vals.push_back(X_[i][feature_idx]);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    return vals;
}

} // namespace rf
