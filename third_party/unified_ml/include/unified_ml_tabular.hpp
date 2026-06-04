#pragma once

#include "unified_ml_artifact.hpp"
#include "unified_ml_metrics.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace unified_ml {

enum class TabularModelKind {
    RandomForest,
    SVM,
    XGBoost,
};

struct RandomForestSpec {
    rf::RandomForestParams params{};
};

struct SVMSpec {
    svm::SVMParams params{};
};

struct XGBoostSpec {
    std::string task = "regression";
    int n_estimators = 200;
    float learning_rate = 0.1f;
    int max_depth = 5;
    float subsample = 1.0f;
    float colsample = 1.0f;
    float lambda = 50.0f;
    float gamma = 0.0f;
    std::string grow_policy = "depthwise";
    int num_class = 2;
};

using TabularSpec = std::variant<RandomForestSpec, SVMSpec, XGBoostSpec>;

struct FitSummary {
    TabularModelKind model_kind = TabularModelKind::RandomForest;
    LearningTask task = LearningTask::Unknown;
    std::size_t rows = 0;
    std::size_t cols = 0;
    ModelCapabilities capabilities{};
};

struct PredictionSummary {
    InferenceOutput output{};
    EvaluationSummary evaluation{};
    bool has_evaluation = false;
};

class TabularModel {
public:
    explicit TabularModel(RandomForestSpec spec)
        : kind_(TabularModelKind::RandomForest), spec_(std::move(spec)) {}
    explicit TabularModel(SVMSpec spec)
        : kind_(TabularModelKind::SVM), spec_(std::move(spec)) {}
    explicit TabularModel(XGBoostSpec spec)
        : kind_(TabularModelKind::XGBoost), spec_(std::move(spec)) {}

    [[nodiscard]] TabularModelKind kind() const noexcept { return kind_; }

    [[nodiscard]] bool is_fitted() const noexcept { return artifact_.has_value(); }
    [[nodiscard]] const ModelArtifact* artifact() const noexcept { return artifact_ ? &*artifact_ : nullptr; }

    [[nodiscard]] ModelCapabilities capabilities() const noexcept {
        if (artifact_) {
            return artifact_->capabilities();
        }

        return std::visit([](const auto& spec) -> ModelCapabilities {
            using Spec = std::decay_t<decltype(spec)>;
            if constexpr (std::is_same_v<Spec, RandomForestSpec>) {
                return capabilities_of<rf::RandomForest>();
            } else if constexpr (std::is_same_v<Spec, SVMSpec>) {
                return capabilities_of<svm::SVM>();
            } else {
                return capabilities_of<xgb::XGBModel>();
            }
        }, spec_);
    }

    [[nodiscard]] FitSummary fit(const DatasetView& dataset) {
        if (!dataset.has_targets()) {
            throw std::invalid_argument("TabularModel::fit: targets are required");
        }
        if (dataset.task() != LearningTask::Classification && dataset.task() != LearningTask::Regression) {
            throw std::invalid_argument("TabularModel::fit: unified tabular facade currently supports classification and regression only");
        }

        artifact_ = std::visit([&](const auto& spec) -> ModelArtifact {
            using Spec = std::decay_t<decltype(spec)>;
            if constexpr (std::is_same_v<Spec, RandomForestSpec>) {
                auto model = rf::RandomForest(spec.params);
                const auto rf_task = dataset.task() == LearningTask::Regression ? rf::TaskType::Regression : rf::TaskType::Classification;
                model.fit(dataset.to_rf_dataset(rf_task));
                return ModelArtifact(std::move(model));
            } else if constexpr (std::is_same_v<Spec, SVMSpec>) {
                auto model = svm::SVM(spec.params);
                const auto features = DatasetView::to_nested_vectors(dataset.features());
                if (dataset.task() == LearningTask::Regression) {
                    model.fit_regression(features, dataset.targets());
                } else {
                    std::vector<int> labels;
                    labels.reserve(dataset.targets().size());
                    for (double value : dataset.targets()) {
                        labels.push_back(static_cast<int>(value));
                    }
                    model.fit(features, labels);
                }
                return ModelArtifact(std::move(model));
            } else {
                xgb::XGBModel model(spec.task);
                model.n_estimators(spec.n_estimators)
                     .learning_rate(spec.learning_rate)
                     .max_depth(spec.max_depth)
                     .subsample(spec.subsample)
                     .colsample(spec.colsample)
                     .lambda(spec.lambda)
                     .gamma(spec.gamma)
                     .grow_policy(spec.grow_policy)
                     .num_class(spec.num_class);

                std::vector<xgb::bst_float> labels;
                labels.reserve(dataset.targets().size());
                for (double value : dataset.targets()) {
                    labels.push_back(static_cast<xgb::bst_float>(value));
                }
                model.fit(dataset.features(), labels);
                return ModelArtifact(std::move(model));
            }
        }, spec_);

        FitSummary summary;
        summary.model_kind = kind_;
        summary.task = dataset.task();
        summary.rows = dataset.rows();
        summary.cols = dataset.cols();
        summary.capabilities = artifact_->capabilities();
        return summary;
    }

    [[nodiscard]] PredictionSummary predict(const DatasetView& dataset, int n_classes = 0) const {
        if (!artifact_) {
            throw std::runtime_error("TabularModel::predict: model is not fitted");
        }

        PredictionSummary summary;
        summary.output = artifact_->run(dataset);
        if (dataset.has_targets()) {
            summary.evaluation = evaluate(dataset, summary.output.values, n_classes);
            summary.has_evaluation = true;
        }
        return summary;
    }

    void save(const std::string& path, ArtifactFormat format = ArtifactFormat::Native) const {
        if (!artifact_) {
            throw std::runtime_error("TabularModel::save: model is not fitted");
        }
        artifact_->save(path, format);
    }

    void export_artifact(const std::string& path, ArtifactFormat format = ArtifactFormat::Fast) const {
        if (!artifact_) {
            throw std::runtime_error("TabularModel::export_artifact: model is not fitted");
        }
        artifact_->export_artifact(path, format);
    }

    [[nodiscard]] static TabularModel load(TabularModelKind kind,
                                           const std::string& path,
                                           ArtifactFormat format = ArtifactFormat::Native) {
        TabularModel model = [&]() {
            switch (kind) {
                case TabularModelKind::RandomForest:
                    return TabularModel(RandomForestSpec{});
                case TabularModelKind::SVM:
                    return TabularModel(SVMSpec{});
                case TabularModelKind::XGBoost:
                    return TabularModel(XGBoostSpec{});
            }
            throw std::invalid_argument("TabularModel::load: unsupported kind");
        }();

        model.artifact_ = ModelArtifact::load(to_model_kind(kind), path, format);
        return model;
    }

private:
    static ModelKind to_model_kind(TabularModelKind kind) {
        switch (kind) {
            case TabularModelKind::RandomForest: return ModelKind::RandomForest;
            case TabularModelKind::SVM: return ModelKind::SVM;
            case TabularModelKind::XGBoost: return ModelKind::XGBoost;
        }
        throw std::invalid_argument("TabularModel::to_model_kind: unsupported kind");
    }

    TabularModelKind kind_;
    TabularSpec spec_;
    std::optional<ModelArtifact> artifact_;
};

} // namespace unified_ml
