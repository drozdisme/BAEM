#pragma once

#include "core/linalg.hpp"
#include "core/matrix_view.hpp"
#include "models/rf/dataset.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace unified_ml {

enum class LearningTask {
    Classification,
    Regression,
    Clustering,
    AnomalyDetection,
    DimensionalityReduction,
    SequenceModeling,
    OperatorLearning,
    SymbolicRegression,
    Unknown,
};

class DatasetView {
public:
    DatasetView() = default;

    DatasetView(core::MatrixView features,
                std::vector<double> targets = {},
                LearningTask task = LearningTask::Unknown)
        : features_(features),
          targets_(std::move(targets)),
          task_(task) {
        validate();
    }

    DatasetView(const core::Matrix& features,
                std::vector<double> targets = {},
                LearningTask task = LearningTask::Unknown)
        : owned_features_(features),
          features_(owned_features_.storage(), owned_features_.rows(), owned_features_.cols(), owned_features_.cols()),
          targets_(std::move(targets)),
          task_(task) {
        validate();
    }

    DatasetView(std::vector<std::vector<double>> features,
                std::vector<double> targets = {},
                LearningTask task = LearningTask::Unknown)
        : owned_features_(to_matrix(features)),
          features_(owned_features_.storage(), owned_features_.rows(), owned_features_.cols(), owned_features_.cols()),
          targets_(std::move(targets)),
          task_(task) {
        validate();
    }

    [[nodiscard]] core::MatrixView features() const noexcept { return features_; }
    [[nodiscard]] const std::vector<double>& targets() const noexcept { return targets_; }
    [[nodiscard]] LearningTask task() const noexcept { return task_; }
    [[nodiscard]] bool has_targets() const noexcept { return !targets_.empty(); }
    [[nodiscard]] std::size_t rows() const noexcept { return features_.rows; }
    [[nodiscard]] std::size_t cols() const noexcept { return features_.cols; }

    [[nodiscard]] rf::Dataset to_rf_dataset(rf::TaskType task_type) const {
        if (!has_targets()) {
            throw std::invalid_argument("DatasetView::to_rf_dataset: targets are required");
        }
        return rf::Dataset(to_nested_vectors(features_), targets_, task_type);
    }

    [[nodiscard]] static DatasetView from_rf_dataset(const rf::Dataset& dataset) {
        LearningTask task = dataset.task() == rf::TaskType::Classification
            ? LearningTask::Classification
            : LearningTask::Regression;
        return DatasetView(dataset.X_view(), dataset.y(), task);
    }

    [[nodiscard]] static core::Matrix to_matrix(const std::vector<std::vector<double>>& rows) {
        if (rows.empty()) {
            return {};
        }
        const std::size_t cols = rows.front().size();
        for (const auto& row : rows) {
            if (row.size() != cols) {
                throw std::invalid_argument("DatasetView: inconsistent row width");
            }
        }

        core::Matrix matrix(rows.size(), cols, 0.0);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                matrix(r, c) = rows[r][c];
            }
        }
        return matrix;
    }

    [[nodiscard]] static std::vector<std::vector<double>> to_nested_vectors(core::MatrixView view) {
        std::vector<std::vector<double>> rows(view.rows, std::vector<double>(view.cols, 0.0));
        for (std::size_t r = 0; r < view.rows; ++r) {
            for (std::size_t c = 0; c < view.cols; ++c) {
                rows[r][c] = view(r, c);
            }
        }
        return rows;
    }

private:
    void validate() const {
        if (!targets_.empty() && targets_.size() != features_.rows) {
            throw std::invalid_argument("DatasetView: targets size must match number of rows");
        }
    }

    core::Matrix owned_features_{};
    core::MatrixView features_{};
    std::vector<double> targets_{};
    LearningTask task_{LearningTask::Unknown};
};

} // namespace unified_ml
