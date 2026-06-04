#pragma once

#include "models/dbscan/dbscan.hpp"
#include "models/gp/gaussian_process.hpp"
#include "models/iforest/isolation_forest.hpp"
#include "models/pca/pca.hpp"
#include "models/sindy/sindy.hpp"
#include "unified_ml_capabilities.hpp"
#include "unified_ml_dataset.hpp"
#include "unified_ml_phase2_artifact.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace unified_ml {

enum class AdvancedModelKind {
    GaussianProcess,
    PCA,
    IsolationForest,
    DBSCAN,
    SINDy,
};

struct GPSpec {
    gp::GPParams params{};
};

struct PCASpec {
    std::size_t n_components = 0;
};

struct IsolationForestSpec {
    int n_trees = 100;
    int subsample_size = 256;
    int max_height = -1;
    std::uint64_t seed = 42;
};

struct DBSCANSpec {
    double epsilon = 0.5;
    int min_points = 5;
};

struct SINDySpec {
    sindy::SINDyParams params{};
};

struct AdvancedFitSummary {
    AdvancedModelKind model_kind = AdvancedModelKind::GaussianProcess;
    LearningTask task = LearningTask::Unknown;
    std::size_t rows = 0;
    std::size_t cols = 0;
    ModelCapabilities capabilities{};
};

struct AdvancedPredictionSummary {
    std::vector<double> values;
    std::vector<std::vector<double>> matrix_values;
    core::Matrix transformed;
    std::vector<std::string> text;
};

class AdvancedModel {
public:
    using SpecVariant = std::variant<GPSpec, PCASpec, IsolationForestSpec, DBSCANSpec, SINDySpec>;
    using ModelVariant = std::variant<gp::GaussianProcessRegressor, pca::PCA, iforest::IsolationForest, dbscan::DBSCAN, sindy::SINDy>;

    explicit AdvancedModel(GPSpec spec) : kind_(AdvancedModelKind::GaussianProcess), spec_(std::move(spec)) {}
    explicit AdvancedModel(PCASpec spec) : kind_(AdvancedModelKind::PCA), spec_(std::move(spec)) {}
    explicit AdvancedModel(IsolationForestSpec spec) : kind_(AdvancedModelKind::IsolationForest), spec_(std::move(spec)) {}
    explicit AdvancedModel(DBSCANSpec spec) : kind_(AdvancedModelKind::DBSCAN), spec_(std::move(spec)) {}
    explicit AdvancedModel(SINDySpec spec) : kind_(AdvancedModelKind::SINDy), spec_(std::move(spec)) {}

    [[nodiscard]] bool is_fitted() const noexcept { return model_.has_value(); }
    [[nodiscard]] AdvancedModelKind kind() const noexcept { return kind_; }

    void save(const std::string& path, AdvancedArtifactFormat format = AdvancedArtifactFormat::Native) const {
        if (!model_) throw std::runtime_error("AdvancedModel::save: model is not fitted");
        if (format != AdvancedArtifactFormat::Native) {
            throw std::invalid_argument("AdvancedModel::save: unsupported format");
        }
        std::visit([&](const auto& model) {
            using Model = std::decay_t<decltype(model)>;
            if constexpr (std::is_same_v<Model, dbscan::DBSCAN>) {
                throw std::invalid_argument("AdvancedModel::save(DBSCAN): persistence is not implemented yet");
            } else {
                model.save(path);
            }
        }, *model_);
    }

    [[nodiscard]] static AdvancedModel load(AdvancedModelKind kind,
                                            const std::string& path,
                                            AdvancedArtifactFormat format = AdvancedArtifactFormat::Native) {
        if (format != AdvancedArtifactFormat::Native) {
            throw std::invalid_argument("AdvancedModel::load: unsupported format");
        }
        switch (kind) {
            case AdvancedModelKind::GaussianProcess: {
                AdvancedModel model(GPSpec{});
                model.model_ = gp::GaussianProcessRegressor::load(path);
                return model;
            }
            case AdvancedModelKind::PCA: {
                AdvancedModel model(PCASpec{});
                model.model_ = pca::PCA::load(path);
                return model;
            }
            case AdvancedModelKind::IsolationForest: {
                AdvancedModel model(IsolationForestSpec{});
                model.model_ = iforest::IsolationForest::load(path);
                return model;
            }
            case AdvancedModelKind::SINDy: {
                AdvancedModel model(SINDySpec{});
                model.model_ = sindy::SINDy::load(path);
                return model;
            }
            case AdvancedModelKind::DBSCAN:
                throw std::invalid_argument("AdvancedModel::load(DBSCAN): persistence is not implemented yet");
        }
        throw std::invalid_argument("AdvancedModel::load: unsupported kind");
    }

    [[nodiscard]] AdvancedFitSummary fit(const DatasetView& dataset) {
        model_ = std::visit([&](const auto& spec) -> ModelVariant {
            using Spec = std::decay_t<decltype(spec)>;
            const auto rows = DatasetView::to_nested_vectors(dataset.features());
            if constexpr (std::is_same_v<Spec, GPSpec>) {
                if (!dataset.has_targets()) throw std::invalid_argument("AdvancedModel(GP)::fit requires targets");
                gp::GaussianProcessRegressor model(spec.params);
                model.fit(rows, dataset.targets());
                return model;
            } else if constexpr (std::is_same_v<Spec, PCASpec>) {
                pca::PCA model(spec.n_components);
                model.fit(dataset.features());
                return model;
            } else if constexpr (std::is_same_v<Spec, IsolationForestSpec>) {
                iforest::IsolationForest model(spec.n_trees, spec.subsample_size, spec.max_height, spec.seed);
                model.fit(rows);
                return model;
            } else if constexpr (std::is_same_v<Spec, DBSCANSpec>) {
                std::vector<dbscan::Point> points;
                points.reserve(dataset.rows());
                for (std::size_t r = 0; r < dataset.rows(); ++r) {
                    std::vector<double> coords(dataset.cols(), 0.0);
                    for (std::size_t c = 0; c < dataset.cols(); ++c) coords[c] = dataset.features()(r, c);
                    points.emplace_back(coords);
                }
                dbscan::DBSCAN model(spec.epsilon, spec.min_points);
                (void)model.fit(points);
                return model;
            } else {
                if (!dataset.has_targets()) throw std::invalid_argument("AdvancedModel(SINDy)::fit requires derivative targets or trajectory-specific API");
                throw std::invalid_argument("AdvancedModel(SINDy)::fit is not mapped from DatasetView yet, use low-level SINDy API for derivative-aware fitting");
            }
        }, spec_);

        AdvancedFitSummary summary;
        summary.model_kind = kind_;
        summary.task = dataset.task();
        summary.rows = dataset.rows();
        summary.cols = dataset.cols();
        summary.capabilities = std::visit([](const auto& model) {
            return capabilities_of(model);
        }, *model_);
        return summary;
    }

    [[nodiscard]] AdvancedPredictionSummary run(const DatasetView& dataset) {
        if (!model_) throw std::runtime_error("AdvancedModel::run: model is not fitted");
        AdvancedPredictionSummary summary;
        const auto rows = DatasetView::to_nested_vectors(dataset.features());

        std::visit([&](auto& model) {
            using Model = std::decay_t<decltype(model)>;
            if constexpr (std::is_same_v<Model, gp::GaussianProcessRegressor>) {
                summary.values = model.predict(rows);
            } else if constexpr (std::is_same_v<Model, pca::PCA>) {
                summary.transformed = model.transform(dataset.features());
            } else if constexpr (std::is_same_v<Model, iforest::IsolationForest>) {
                summary.values = model.score_batch(rows);
            } else if constexpr (std::is_same_v<Model, dbscan::DBSCAN>) {
                std::vector<dbscan::Point> points;
                points.reserve(dataset.rows());
                for (std::size_t r = 0; r < dataset.rows(); ++r) {
                    std::vector<double> coords(dataset.cols(), 0.0);
                    for (std::size_t c = 0; c < dataset.cols(); ++c) coords[c] = dataset.features()(r, c);
                    points.emplace_back(coords);
                }
                const auto labels = model.fit(points);
                summary.values.reserve(labels.size());
                for (int label : labels) summary.values.push_back(static_cast<double>(label));
            } else if constexpr (std::is_same_v<Model, sindy::SINDy>) {
                summary.matrix_values = model.predict_derivative(rows);
                summary.text = model.equations();
            }
        }, *model_);

        return summary;
    }

private:
    AdvancedModelKind kind_;
    SpecVariant spec_;
    std::optional<ModelVariant> model_;
};

} // namespace unified_ml
