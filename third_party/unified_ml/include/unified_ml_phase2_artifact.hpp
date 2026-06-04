#pragma once

#include "models/gp/gaussian_process.hpp"
#include "models/iforest/isolation_forest.hpp"
#include "models/pca/pca.hpp"
#include "models/sindy/sindy.hpp"
#include "unified_ml_capabilities.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace unified_ml {

enum class AdvancedArtifactFormat {
    Native,
};

enum class AdvancedArtifactKind {
    GaussianProcess,
    PCA,
    IsolationForest,
    SINDy,
};

class AdvancedModelArtifact {
public:
    using ModelVariant = std::variant<gp::GaussianProcessRegressor, pca::PCA, iforest::IsolationForest, sindy::SINDy>;

    explicit AdvancedModelArtifact(gp::GaussianProcessRegressor model) : model_(std::move(model)) {}
    explicit AdvancedModelArtifact(pca::PCA model) : model_(std::move(model)) {}
    explicit AdvancedModelArtifact(iforest::IsolationForest model) : model_(std::move(model)) {}
    explicit AdvancedModelArtifact(sindy::SINDy model) : model_(std::move(model)) {}

    [[nodiscard]] ModelCapabilities capabilities() const noexcept {
        return std::visit([](const auto& model) {
            return capabilities_of(model);
        }, model_);
    }

    [[nodiscard]] const ModelVariant& variant() const noexcept { return model_; }

    void save(const std::string& path, AdvancedArtifactFormat format = AdvancedArtifactFormat::Native) const {
        if (format != AdvancedArtifactFormat::Native) {
            throw std::invalid_argument("AdvancedModelArtifact::save: unsupported format");
        }
        std::visit([&](const auto& model) {
            save_impl(model, path);
        }, model_);
    }

    [[nodiscard]] static AdvancedModelArtifact load(AdvancedArtifactKind kind,
                                                    const std::string& path,
                                                    AdvancedArtifactFormat format = AdvancedArtifactFormat::Native) {
        if (format != AdvancedArtifactFormat::Native) {
            throw std::invalid_argument("AdvancedModelArtifact::load: unsupported format");
        }
        switch (kind) {
            case AdvancedArtifactKind::GaussianProcess:
                return AdvancedModelArtifact(gp::GaussianProcessRegressor::load(path));
            case AdvancedArtifactKind::PCA:
                return AdvancedModelArtifact(pca::PCA::load(path));
            case AdvancedArtifactKind::IsolationForest:
                return AdvancedModelArtifact(iforest::IsolationForest::load(path));
            case AdvancedArtifactKind::SINDy:
                return AdvancedModelArtifact(sindy::SINDy::load(path));
            default:
                throw std::invalid_argument("AdvancedModelArtifact::load: unsupported kind");
        }
        throw std::invalid_argument("AdvancedModelArtifact::load: unsupported kind");
    }

private:
    static void save_impl(const gp::GaussianProcessRegressor& model, const std::string& path) { model.save(path); }
    static void save_impl(const pca::PCA& model, const std::string& path) { model.save(path); }
    static void save_impl(const iforest::IsolationForest& model, const std::string& path) { model.save(path); }
    static void save_impl(const sindy::SINDy& model, const std::string& path) { model.save(path); }

    ModelVariant model_;
};

} // namespace unified_ml
