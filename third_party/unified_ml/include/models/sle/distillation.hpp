#pragma once

#include "autograd/tensor.h"
#include "models/iforest/isolation_forest.hpp"
#include "models/pca/pca.hpp"
#include "models/rf/random_forest.hpp"
#include "models/sle/circuit.hpp"
#include "models/sle/framework.hpp"
#include "models/sle/synthesis.hpp"
#include "models/xgboost/xgboost_enhanced.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace core::models::sle {

struct DistillConfig {
    std::size_t gate_budget = 32;
    double target_accuracy = 0.97;
    double threshold = 0.5;
    SynthesisConfig synthesis{};
};

struct DistillDataset {
    std::vector<TrainingExample> examples;
    std::vector<double> thresholds;
    std::size_t feature_count = 0;
};

class DistillDatasetBuilder {
public:
    explicit DistillDatasetBuilder(DistillConfig config = {});

    [[nodiscard]] DistillDataset build(std::span<const std::vector<double>> features,
                                       std::span<const double> predictions) const;

private:
    [[nodiscard]] double find_optimal_threshold(std::span<const double> values,
                                                std::span<const double> labels) const;
    [[nodiscard]] static double information_gain(std::span<const double> values,
                                                 std::span<const double> labels,
                                                 double threshold);
    [[nodiscard]] static std::vector<std::uint64_t> quantize_feature(std::span<const double> values,
                                                                     double threshold);

    DistillConfig config_{};
};

[[nodiscard]] BooleanCascade distill_to_circuit(const DistillDataset& dataset,
                                                const DistillConfig& config = {});

[[nodiscard]] BooleanCascade distill_to_circuit(const rf::RandomForest& model,
                                                const DistillDataset& dataset,
                                                const DistillConfig& config = {});

[[nodiscard]] BooleanCascade distill_to_circuit(const xgb::XGBModel& model,
                                                const DistillDataset& dataset,
                                                const DistillConfig& config = {});

[[nodiscard]] BooleanCascade distill_to_circuit(const iforest::IsolationForest& model,
                                                const DistillDataset& dataset,
                                                const DistillConfig& config = {});

[[nodiscard]] BooleanCascade distill_to_circuit(const pca::PCA& model,
                                                const DistillDataset& dataset,
                                                const DistillConfig& config = {});

[[nodiscard]] inline BooleanCascade distill_to_circuit(
    const xgb::XGBModel& model,
    std::span<const std::vector<double>> features,
    const DistillConfig& config = {}) {
    DistillDatasetBuilder builder(config);

    std::vector<std::vector<float>> xgb_features;
    xgb_features.reserve(features.size());
    for (const auto& row : features) {
        xgb_features.emplace_back(row.begin(), row.end());
    }

    const auto predictions_f = model.predict(xgb_features);
    std::vector<double> predictions(predictions_f.begin(), predictions_f.end());
    const auto dataset = builder.build(features, predictions);
    return distill_to_circuit(model, dataset, config);
}

template <class Model>
[[nodiscard]] BooleanCascade distill_to_circuit(const Model& model,
                                                std::span<const std::vector<double>> features,
                                                const DistillConfig& config = {}) {
    DistillDatasetBuilder builder(config);
    const auto predictions = model.predict(std::vector<std::vector<double>>(features.begin(), features.end()));
    const auto dataset = builder.build(features, predictions);
    return distill_to_circuit(dataset, config);
}

} // namespace core::models::sle
