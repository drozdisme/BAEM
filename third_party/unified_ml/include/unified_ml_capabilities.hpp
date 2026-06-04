#pragma once

#include "models/dbscan/dbscan.hpp"
#include "models/deep_onet/model.hpp"
#include "models/gp/gaussian_process.hpp"
#include "models/iforest/isolation_forest.hpp"
#include "models/mlp/model.hpp"
#include "models/pca/pca.hpp"
#include "models/pideeponet/pideeponet.hpp"
#include "models/pinn/pinn_model.hpp"
#include "models/rf/random_forest.hpp"
#include "models/sindy/sindy.hpp"
#include "models/svm/svm.hpp"
#include "models/transformer/transformer_block.hpp"
#include "models/xgboost/xgboost_enhanced.hpp"
#include "models/sle/full_engine.hpp"

namespace unified_ml {

struct ModelCapabilities {
    bool supports_classification = false;
    bool supports_regression = false;
    bool supports_online_inference = false;
    bool supports_batch_inference = false;
    bool supports_serialization = false;
    bool supports_fast_artifact_export = false;
    bool supports_exact_distillation = false;
    bool supports_constraints = false;
};

template <typename Model>
inline constexpr ModelCapabilities capability_descriptor_v{};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<mlp::Model>{
    true, true, true, true, false, false, true, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<rf::RandomForest>{
    true, true, true, true, true, false, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<svm::SVM>{
    true, true, true, true, true, false, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<xgb::XGBModel>{
    true, true, true, true, true, true, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<xgb::XGBRegressor>{
    false, true, true, true, true, true, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<xgb::XGBClassifier>{
    true, false, true, true, true, true, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<xgb::XGBTweedieRegressor>{
    false, true, true, true, true, true, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<gp::GaussianProcessRegressor>{
    true, true, true, true, true, false, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<pca::PCA>{
    false, false, true, true, true, false, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<iforest::IsolationForest>{
    false, false, true, true, true, false, true, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<dbscan::DBSCAN>{
    false, false, false, true, false, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<sindy::SINDy>{
    false, true, true, true, true, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<pinn::PINNModel>{
    false, true, false, true, false, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<deep_onet::DeepONet>{
    false, true, false, true, false, false, false, false,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<pideeponet::PIDeepONet>{
    false, true, false, true, true, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<transformer::TransformerBlock>{
    false, false, true, true, false, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<transformer::TransformerEncoder>{
    true, false, true, true, false, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<transformer::TransformerSeq2Seq>{
    false, false, true, true, false, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<transformer::TransformerSystem>{
    true, false, true, true, false, false, false, true,
};

template <>
inline constexpr ModelCapabilities capability_descriptor_v<core::models::sle::FullEngine>{
    true, true, true, true, false, true, false, true,
};

template <typename Model>
struct capability_descriptor {
    static constexpr ModelCapabilities value = capability_descriptor_v<Model>;
};

template <typename Model>
[[nodiscard]] constexpr ModelCapabilities capabilities_of() noexcept {
    return capability_descriptor_v<Model>;
}

[[nodiscard]] constexpr ModelCapabilities capabilities_of(const mlp::Model&) noexcept { return capabilities_of<mlp::Model>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const rf::RandomForest&) noexcept { return capabilities_of<rf::RandomForest>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const svm::SVM&) noexcept { return capabilities_of<svm::SVM>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const xgb::XGBModel&) noexcept { return capabilities_of<xgb::XGBModel>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const xgb::XGBRegressor&) noexcept { return capabilities_of<xgb::XGBRegressor>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const xgb::XGBClassifier&) noexcept { return capabilities_of<xgb::XGBClassifier>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const xgb::XGBTweedieRegressor&) noexcept { return capabilities_of<xgb::XGBTweedieRegressor>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const gp::GaussianProcessRegressor&) noexcept { return capabilities_of<gp::GaussianProcessRegressor>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const pca::PCA&) noexcept { return capabilities_of<pca::PCA>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const iforest::IsolationForest&) noexcept { return capabilities_of<iforest::IsolationForest>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const dbscan::DBSCAN&) noexcept { return capabilities_of<dbscan::DBSCAN>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const sindy::SINDy&) noexcept { return capabilities_of<sindy::SINDy>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const pinn::PINNModel&) noexcept { return capabilities_of<pinn::PINNModel>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const deep_onet::DeepONet&) noexcept { return capabilities_of<deep_onet::DeepONet>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const pideeponet::PIDeepONet&) noexcept { return capabilities_of<pideeponet::PIDeepONet>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const transformer::TransformerBlock&) noexcept { return capabilities_of<transformer::TransformerBlock>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const transformer::TransformerEncoder&) noexcept { return capabilities_of<transformer::TransformerEncoder>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const transformer::TransformerSeq2Seq&) noexcept { return capabilities_of<transformer::TransformerSeq2Seq>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const transformer::TransformerSystem&) noexcept { return capabilities_of<transformer::TransformerSystem>(); }
[[nodiscard]] constexpr ModelCapabilities capabilities_of(const core::models::sle::FullEngine&) noexcept { return capabilities_of<core::models::sle::FullEngine>(); }

} // namespace unified_ml
