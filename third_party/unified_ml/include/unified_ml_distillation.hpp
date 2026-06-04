#pragma once

#include "models/iforest/isolation_forest.hpp"
#include "models/mlp/model.hpp"
#include "models/pca/pca.hpp"
#include "models/pinn/neural_network.hpp"
#include "models/rf/random_forest.hpp"
#include "models/sle/distillation.hpp"
#include "models/xgboost/xgboost_enhanced.hpp"
#include "sle_backend.hpp"
#include "unified_ml_artifact.hpp"
#include "unified_ml_capabilities.hpp"
#include "unified_ml_dataset.hpp"

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace unified_ml {

using DistillationConfig = core::models::sle::DistillConfig;
using DistillationDataset = core::models::sle::DistillDataset;
using DistillationDatasetBuilder = core::models::sle::DistillDatasetBuilder;
using DistilledCircuit = core::models::sle::BooleanCascade;

struct DistillationSummary {
    DistilledCircuit circuit{};
    std::size_t gate_count = 0;
    bool exact = false;
};

[[nodiscard]] inline DistillationSummary distill_to_sle(const ModelArtifact& artifact,
                                                        const DatasetView& dataset,
                                                        const DistillationConfig& config = {}) {
    const auto features = DatasetView::to_nested_vectors(dataset.features());

    try {
        auto circuit = std::visit([&](const auto& model) -> DistilledCircuit {
            using Model = std::decay_t<decltype(model)>;
            if constexpr (std::is_same_v<Model, rf::RandomForest> ||
                          std::is_same_v<Model, iforest::IsolationForest>) {
                return core::models::sle::distill_to_circuit(model, std::span<const std::vector<double>>(features.data(), features.size()), config);
            } else if constexpr (std::is_same_v<Model, xgb::XGBModel>) {
                return core::models::sle::distill_to_circuit(model, std::span<const std::vector<double>>(features.data(), features.size()), config);
            } else if constexpr (std::is_same_v<Model, svm::SVM>) {
                throw std::invalid_argument("distill_to_sle(SVM): no SLE distillation path is implemented yet");
            } else {
                throw std::invalid_argument("distill_to_sle: unsupported artifact model type");
            }
        }, artifact.variant());

        return DistillationSummary{circuit, circuit.gate_count(), false};
    } catch (const std::out_of_range& e) {
        throw std::runtime_error(std::string("distill_to_sle: SLE backend rejected synthesized circuit: ") + e.what());
    }
}

template <typename Model>
[[nodiscard]] inline DistillationSummary distill_to_sle(const Model& model,
                                                        const DatasetView& dataset,
                                                        const DistillationConfig& config = {}) {
    if constexpr (std::is_same_v<Model, rf::RandomForest> ||
                  std::is_same_v<Model, xgb::XGBModel> ||
                  std::is_same_v<Model, iforest::IsolationForest>) {
        const auto features = DatasetView::to_nested_vectors(dataset.features());
        auto circuit = core::models::sle::distill_to_circuit(model,
                                                             std::span<const std::vector<double>>(features.data(), features.size()),
                                                             config);
        return DistillationSummary{circuit, circuit.gate_count(), false};
    } else {
        throw std::invalid_argument("distill_to_sle(model, dataset): unsupported model type for dataset-based SLE distillation");
    }
}

[[nodiscard]] inline DistillationSummary distill_to_sle(const TabularModel& model,
                                                        const DatasetView& dataset,
                                                        const DistillationConfig& config = {}) {
    const auto* artifact = model.artifact();
    if (artifact == nullptr) {
        throw std::runtime_error("distill_to_sle(TabularModel): model is not fitted");
    }
    return distill_to_sle(*artifact, dataset, config);
}

[[nodiscard]] inline DistillationSummary distill_to_sle(const mlp::Model& model) {
    auto circuit = sle_backend::distill_to_logic(model);
    return DistillationSummary{circuit, circuit.gate_count(), true};
}

[[nodiscard]] inline DistillationSummary distill_to_sle(const pinn::NeuralNetwork& model) {
    auto circuit = sle_backend::distill_to_logic(model);
    return DistillationSummary{circuit, circuit.gate_count(), true};
}

} // namespace unified_ml
