#pragma once

#include "unified_ml_artifact.hpp"
#include "unified_ml_tabular.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace unified_ml {

struct FeatureImportanceEntry {
    std::string name;
    double value = 0.0;
};

struct ExplainSummary {
    std::vector<FeatureImportanceEntry> feature_importance;
    std::vector<std::vector<double>> attributions;
    std::vector<double> baseline;
    bool has_feature_importance = false;
    bool has_attributions = false;
};

[[nodiscard]] inline ExplainSummary explain(const ModelArtifact& artifact,
                                            const DatasetView* dataset = nullptr) {
    ExplainSummary summary;

    std::visit([&](const auto& model) {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, rf::RandomForest>) {
            const auto values = model.feature_importances();
            summary.feature_importance.reserve(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                summary.feature_importance.push_back({"f" + std::to_string(i), values[i]});
            }
            summary.has_feature_importance = !summary.feature_importance.empty();
        } else if constexpr (std::is_same_v<Model, xgb::XGBModel>) {
            const auto values = model.feature_importances();
            summary.feature_importance.reserve(values.size());
            for (const auto& [name, value] : values) {
                summary.feature_importance.push_back({name, value});
            }
            summary.has_feature_importance = !summary.feature_importance.empty();
            if (dataset != nullptr) {
                const auto shap = model.explain(dataset->features());
                summary.attributions = shap.values;
                summary.baseline = shap.bias;
                summary.has_attributions = !summary.attributions.empty();
            }
        } else {
            // No unified explain contract yet for this backend.
        }
    }, artifact.variant());

    return summary;
}

[[nodiscard]] inline ExplainSummary explain(const TabularModel& model,
                                            const DatasetView* dataset = nullptr) {
    const auto* artifact = model.artifact();
    if (artifact == nullptr) {
        throw std::runtime_error("unified_ml::explain(TabularModel): model is not fitted");
    }
    return explain(*artifact, dataset);
}

} // namespace unified_ml
