#pragma once

#include "models/rf/metrics.hpp"
#include "unified_ml_dataset.hpp"

#include <stdexcept>
#include <vector>

namespace unified_ml {

enum class MetricTask {
    Classification,
    Regression,
};

struct ClassificationMetrics {
    double accuracy = 0.0;
    double macro_f1 = 0.0;
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<std::vector<int>> confusion_matrix;
};

struct RegressionMetrics {
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
};

struct EvaluationSummary {
    MetricTask task = MetricTask::Classification;
    ClassificationMetrics classification{};
    RegressionMetrics regression{};
};

[[nodiscard]] inline EvaluationSummary evaluate_classification(const std::vector<double>& y_true,
                                                               const std::vector<double>& y_pred,
                                                               int n_classes) {
    const auto report = rf::classification_report(y_true, y_pred, n_classes);
    EvaluationSummary summary;
    summary.task = MetricTask::Classification;
    summary.classification.accuracy = report.accuracy;
    summary.classification.macro_f1 = report.macro_f1;
    summary.classification.precision = report.precision;
    summary.classification.recall = report.recall;
    summary.classification.confusion_matrix = rf::confusion_matrix(y_true, y_pred, n_classes);
    return summary;
}

[[nodiscard]] inline EvaluationSummary evaluate_regression(const std::vector<double>& y_true,
                                                           const std::vector<double>& y_pred) {
    const auto report = rf::regression_report(y_true, y_pred);
    EvaluationSummary summary;
    summary.task = MetricTask::Regression;
    summary.regression.mse = report.mse;
    summary.regression.rmse = report.rmse;
    summary.regression.mae = report.mae;
    summary.regression.r2 = report.r2;
    return summary;
}

[[nodiscard]] inline EvaluationSummary evaluate(const DatasetView& dataset,
                                                const std::vector<double>& predictions,
                                                int n_classes = 0) {
    if (!dataset.has_targets()) {
        throw std::invalid_argument("unified_ml::evaluate: dataset targets are required");
    }

    switch (dataset.task()) {
        case LearningTask::Classification:
            if (n_classes <= 0) {
                throw std::invalid_argument("unified_ml::evaluate: n_classes must be positive for classification");
            }
            return evaluate_classification(dataset.targets(), predictions, n_classes);
        case LearningTask::Regression:
            return evaluate_regression(dataset.targets(), predictions);
        default:
            throw std::invalid_argument("unified_ml::evaluate: dataset task is not supported by the unified metrics facade yet");
    }
}

} // namespace unified_ml
