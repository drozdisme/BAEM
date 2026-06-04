#pragma once

#include "models/rf/random_forest.hpp"
#include "models/svm/svm.hpp"
#include "models/xgboost/xgboost_enhanced.hpp"
#include "unified_ml_capabilities.hpp"
#include "unified_ml_dataset.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace unified_ml {

enum class ArtifactFormat {
    Native,
    Binary,
    Fast,
};

enum class ModelKind {
    RandomForest,
    SVM,
    XGBoost,
};

struct InferenceOutput {
    std::vector<double> values;
    std::vector<std::vector<double>> probabilities;
    std::vector<double> scores;

    [[nodiscard]] bool has_probabilities() const noexcept { return !probabilities.empty(); }
    [[nodiscard]] bool has_scores() const noexcept { return !scores.empty(); }
};

class ModelArtifact {
public:
    using ModelVariant = std::variant<rf::RandomForest, svm::SVM, xgb::XGBModel>;

    explicit ModelArtifact(rf::RandomForest model)
        : kind_(ModelKind::RandomForest), model_(std::move(model)) {}
    explicit ModelArtifact(svm::SVM model)
        : kind_(ModelKind::SVM), model_(std::move(model)) {}
    explicit ModelArtifact(xgb::XGBModel model)
        : kind_(ModelKind::XGBoost), model_(std::move(model)) {}

    [[nodiscard]] ModelKind kind() const noexcept { return kind_; }

    [[nodiscard]] ModelCapabilities capabilities() const noexcept {
        return std::visit([](const auto& model) {
            return unified_ml::capabilities_of(model);
        }, model_);
    }

    [[nodiscard]] const ModelVariant& variant() const noexcept { return model_; }
    [[nodiscard]] ModelVariant& variant() noexcept { return model_; }

    void save(const std::string& path, ArtifactFormat format = ArtifactFormat::Native) const {
        std::visit([&](const auto& model) {
            save_impl(model, path, format);
        }, model_);
    }

    void export_artifact(const std::string& path, ArtifactFormat format = ArtifactFormat::Fast) const {
        std::visit([&](const auto& model) {
            export_impl(model, path, format);
        }, model_);
    }

    [[nodiscard]] InferenceOutput run(const DatasetView& dataset) const {
        return std::visit([&](const auto& model) {
            return run_impl(model, dataset);
        }, model_);
    }

    [[nodiscard]] static ModelArtifact load(ModelKind kind,
                                            const std::string& path,
                                            ArtifactFormat format = ArtifactFormat::Native) {
        switch (kind) {
            case ModelKind::RandomForest:
                return ModelArtifact(rf::RandomForest::load(path));
            case ModelKind::SVM:
                return ModelArtifact(svm::SVM::load(path));
            case ModelKind::XGBoost: {
                xgb::XGBModel model;
                if (format == ArtifactFormat::Binary) {
                    model.load_binary(path);
                } else {
                    model.load(path);
                }
                return ModelArtifact(std::move(model));
            }
        }
        throw std::invalid_argument("ModelArtifact::load: unsupported model kind");
    }

private:
    static void save_impl(const rf::RandomForest& model,
                          const std::string& path,
                          ArtifactFormat format) {
        if (format != ArtifactFormat::Native && format != ArtifactFormat::Binary) {
            throw std::invalid_argument("ModelArtifact::save(RandomForest): unsupported format");
        }
        model.save(path);
    }

    static void save_impl(const svm::SVM& model,
                          const std::string& path,
                          ArtifactFormat format) {
        if (format != ArtifactFormat::Native && format != ArtifactFormat::Binary) {
            throw std::invalid_argument("ModelArtifact::save(SVM): unsupported format");
        }
        model.save(path);
    }

    static void save_impl(const xgb::XGBModel& model,
                          const std::string& path,
                          ArtifactFormat format) {
        if (format == ArtifactFormat::Binary) {
            model.save_binary(path);
            return;
        }
        model.save(path);
    }

    static void export_impl(const rf::RandomForest& model,
                            const std::string& path,
                            ArtifactFormat format) {
        if (format != ArtifactFormat::Fast && format != ArtifactFormat::Native) {
            throw std::invalid_argument("ModelArtifact::export(RandomForest): unsupported format");
        }
        model.save(path);
    }

    static void export_impl(const svm::SVM& model,
                            const std::string& path,
                            ArtifactFormat format) {
        if (format != ArtifactFormat::Fast && format != ArtifactFormat::Native) {
            throw std::invalid_argument("ModelArtifact::export(SVM): unsupported format");
        }
        model.save(path);
    }

    static void export_impl(const xgb::XGBModel& model,
                            const std::string& path,
                            ArtifactFormat format) {
        if (format == ArtifactFormat::Fast || format == ArtifactFormat::Binary) {
            model.save_binary(path);
            return;
        }
        const std::filesystem::path export_path(path);
        if (export_path.has_extension()) {
            model.save(path);
        } else {
            model.export_dashboard(path);
        }
    }

    static InferenceOutput run_impl(const rf::RandomForest& model,
                                    const DatasetView& dataset) {
        InferenceOutput out;
        const auto rf_task = dataset.task() == LearningTask::Regression
            ? rf::TaskType::Regression
            : rf::TaskType::Classification;
        auto rf_dataset = dataset.to_rf_dataset(rf_task);
        out.values = model.predict(rf_dataset);
        if (rf_task == rf::TaskType::Classification) {
            out.probabilities = model.predict_proba(rf_dataset);
        }
        return out;
    }

    static InferenceOutput run_impl(const svm::SVM& model,
                                    const DatasetView& dataset) {
        InferenceOutput out;
        const auto features = DatasetView::to_nested_vectors(dataset.features());
        if (dataset.task() == LearningTask::Regression) {
            out.values = model.predict_regression(features);
            return out;
        }

        const auto labels = model.predict(features);
        out.values.reserve(labels.size());
        for (int label : labels) {
            out.values.push_back(static_cast<double>(label));
        }
        out.probabilities = model.predict_proba(features);
        return out;
    }

    static InferenceOutput run_impl(const xgb::XGBModel& model,
                                    const DatasetView& dataset) {
        InferenceOutput out;
        std::vector<std::vector<xgb::bst_float>> features(dataset.rows(), std::vector<xgb::bst_float>(dataset.cols(), 0.0f));
        for (std::size_t r = 0; r < dataset.rows(); ++r) {
            for (std::size_t c = 0; c < dataset.cols(); ++c) {
                features[r][c] = static_cast<xgb::bst_float>(dataset.features()(r, c));
            }
        }
        const auto preds = model.predict(features);
        out.values.assign(preds.begin(), preds.end());
        return out;
    }

    ModelKind kind_;
    ModelVariant model_;
};

} // namespace unified_ml
