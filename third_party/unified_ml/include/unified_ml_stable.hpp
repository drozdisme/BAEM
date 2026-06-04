#pragma once

/**
 * @file unified_ml_stable.hpp
 * @brief Stable production-facing umbrella header for the unified_ml SDK.
 *
 * This header aggregates the supported public surface intended for commercial
 * integration. Prefer including this header, or the extensionless
 * `#include <unified_ml>` compatibility entry point, from application code.
 */

#include "unified_ml_version.hpp"
#include "unified_ml_stability.hpp"
#include "unified_ml_capabilities.hpp"
#include "unified_ml_dataset.hpp"
#include "unified_ml_metrics.hpp"
#include "unified_ml_artifact.hpp"
#include "unified_ml_tabular.hpp"
#include "unified_ml_reports.hpp"
#include "unified_ml_phase2.hpp"
#include "unified_ml_phase2_artifact.hpp"
#include "unified_ml_distillation.hpp"
#include "unified_ml_mlp.hpp"

#include "autograd/autograd.h"
#include "core/sdk_common.hpp"
#include "core/linalg.hpp"
#include "core/random.hpp"
#include "core/activations.hpp"
#include "core/optimizers.hpp"
#include "core/quantization.hpp"

#include "models/mlp/layer.hpp"
#include "models/mlp/linear.hpp"
#include "models/mlp/activation.hpp"
#include "models/mlp/loss.hpp"
#include "models/mlp/sequential.hpp"
#include "models/mlp/model.hpp"

#include "models/deep_onet/activations.hpp"
#include "models/deep_onet/layers.hpp"
#include "models/deep_onet/loss.hpp"
#include "models/deep_onet/branch_net.hpp"
#include "models/deep_onet/trunk_net.hpp"
#include "models/deep_onet/optimizer.hpp"
#include "models/deep_onet/model.hpp"
#include "models/deep_onet/utils.hpp"

#include "models/pinn/tensor_wrapper.hpp"
#include "models/pinn/activations.hpp"
#include "models/pinn/layers.hpp"
#include "models/pinn/loss.hpp"
#include "models/pinn/optimizers.hpp"
#include "models/pinn/neural_network.hpp"
#include "models/pinn/pde.hpp"
#include "models/pinn/pinn_model.hpp"
#include "models/pinn/pinn_solver.hpp"
#include "models/pinn/utils.hpp"

#include "models/rf/utils.hpp"
#include "models/rf/dataset.hpp"
#include "models/rf/split.hpp"
#include "models/rf/tree.hpp"
#include "models/rf/cart.hpp"
#include "models/rf/metrics.hpp"
#include "models/rf/random_forest.hpp"

#include "models/xgboost/xgboost_enhanced.hpp"

#include "models/dbscan/point.hpp"
#include "models/dbscan/utils.hpp"
#include "models/dbscan/dbscan.hpp"

#include "models/iforest/node.hpp"
#include "models/iforest/math_utils.hpp"
#include "models/iforest/random_utils.hpp"
#include "models/iforest/itree.hpp"
#include "models/iforest/isolation_forest.hpp"

#include "models/pca/pca.hpp"
#include "models/kriging/kriging.hpp"
#include "models/transformer/transformer_block.hpp"
#include "models/svm/svm.hpp"
#include "models/gp/gaussian_process.hpp"
#include "models/sindy/sindy.hpp"
#include "models/pideeponet/pideeponet.hpp"
#include "models/sle/circuit.hpp"
#include "models/sle/synthesis.hpp"
#include "models/sle/full_engine.hpp"
#include "models/sle/framework.hpp"
#include "models/sle/distillation.hpp"
#include "core/symbolic.hpp"
#include "sle_backend.hpp"

namespace unified_ml::stable {
/** @brief Stability tier for the autograd subsystem. */
inline constexpr stability::Tier autograd_tier = stability::Tier::Core;
/** @brief Stability tier for core math, utility, and memory primitives. */
inline constexpr stability::Tier core_tier = stability::Tier::Core;
/** @brief Stability tier for multilayer perceptron APIs. */
inline constexpr stability::Tier mlp_tier = stability::Tier::Stable;
/** @brief Stability tier for DeepONet APIs. */
inline constexpr stability::Tier deep_onet_tier = stability::Tier::Stable;
/** @brief Stability tier for PINN APIs. */
inline constexpr stability::Tier pinn_tier = stability::Tier::Stable;
/** @brief Stability tier for random forest APIs. */
inline constexpr stability::Tier rf_tier = stability::Tier::Stable;
/** @brief Stability tier for XGBoost-style APIs. */
inline constexpr stability::Tier xgb_tier = stability::Tier::Stable;
/** @brief Stability tier for DBSCAN APIs. */
inline constexpr stability::Tier dbscan_tier = stability::Tier::Stable;
/** @brief Stability tier for isolation forest APIs. */
inline constexpr stability::Tier iforest_tier = stability::Tier::Stable;
/** @brief Stability tier for PCA APIs. */
inline constexpr stability::Tier pca_tier = stability::Tier::Stable;
/** @brief Stability tier for kriging APIs. */
inline constexpr stability::Tier kriging_tier = stability::Tier::Stable;
/** @brief Stability tier for transformer APIs. */
inline constexpr stability::Tier transformer_tier = stability::Tier::Stable;
/** @brief Stability tier for support vector machine APIs. */
inline constexpr stability::Tier svm_tier = stability::Tier::Stable;
/** @brief Stability tier for Gaussian process APIs. */
inline constexpr stability::Tier gp_tier = stability::Tier::Stable;
/** @brief Stability tier for SINDy APIs. */
inline constexpr stability::Tier sindy_tier = stability::Tier::Stable;
/** @brief Stability tier for PI-DeepONet APIs. */
inline constexpr stability::Tier pideeponet_tier = stability::Tier::Stable;
/** @brief Stability tier for SLE APIs. */
inline constexpr stability::Tier sle_tier = stability::Tier::Stable;
} // namespace unified_ml::stable

namespace core {
/** @brief Production namespace alias for core math and utility primitives. */
inline namespace math {}
/** @brief Production namespace alias for compute-oriented internals and kernels. */
namespace hpc = ::core;
} // namespace core

namespace core::models {
/** @brief Stable alias for PCA model APIs. */
namespace pca = ::pca;
/** @brief Stable alias for random forest model APIs. */
namespace random_forest = ::rf;
/** @brief Stable alias for support vector machine APIs. */
namespace svm = ::svm;
/** @brief Stable alias for sparse identification of nonlinear dynamics APIs. */
namespace sindy = ::sindy;
/** @brief Stable alias for Gaussian process APIs. */
namespace gaussian_process = ::gp;
/** @brief Stable alias for DBSCAN clustering APIs. */
namespace dbscan = ::dbscan;
/** @brief Stable alias for isolation forest APIs. */
namespace isolation_forest = ::iforest;
/** @brief Stable alias for multilayer perceptron APIs. */
namespace mlp = ::mlp;
/** @brief Stable alias for DeepONet APIs. */
namespace deep_onet = ::deep_onet;
/** @brief Stable alias for PINN APIs. */
namespace pinn = ::pinn;
/** @brief Stable alias for XGBoost-style APIs. */
namespace xgboost = ::xgb;
/** @brief Stable alias for PI-DeepONet APIs. */
namespace pideeponet = ::pideeponet;
/** @brief Stable alias for transformer APIs. */
namespace transformer = ::transformer;
/** @brief Stable alias for kriging APIs. */
namespace kriging = ::kriging;
/** @brief Stable alias for SLE APIs. */
namespace sle = ::core::models::sle;
} // namespace core::models

namespace unified_ml::stable {
using ::unified_ml::AdvancedArtifactFormat;
using ::unified_ml::AdvancedArtifactKind;
using ::unified_ml::AdvancedFitSummary;
using ::unified_ml::AdvancedModel;
using ::unified_ml::AdvancedModelArtifact;
using ::unified_ml::AdvancedModelKind;
using ::unified_ml::AdvancedPredictionSummary;
using ::unified_ml::ArtifactFormat;
using ::unified_ml::DistillationConfig;
using ::unified_ml::DistillationDataset;
using ::unified_ml::DistillationDatasetBuilder;
using ::unified_ml::DistillationSummary;
using ::unified_ml::DistilledCircuit;
using ::unified_ml::ClassificationMetrics;
using ::unified_ml::DatasetView;
using ::unified_ml::DBSCANSpec;
using ::unified_ml::EvaluationSummary;
using ::unified_ml::ExplainSummary;
using ::unified_ml::MLPActivationKind;
using ::unified_ml::MLPFitSummary;
using ::unified_ml::MLPSpec;
using ::unified_ml::FeatureImportanceEntry;
using ::unified_ml::FitSummary;
using ::unified_ml::GPSpec;
using ::unified_ml::InferenceOutput;
using ::unified_ml::IsolationForestSpec;
using ::unified_ml::LearningTask;
using ::unified_ml::MetricTask;
using ::unified_ml::ModelArtifact;
using ::unified_ml::ModelCapabilities;
using ::unified_ml::ModelKind;
using ::unified_ml::PCASpec;
using ::unified_ml::PredictionSummary;
using ::unified_ml::RandomForestSpec;
using ::unified_ml::RegressionMetrics;
using ::unified_ml::SINDySpec;
using ::unified_ml::SVMSpec;
using ::unified_ml::TabularModel;
using ::unified_ml::TabularModelKind;
using ::unified_ml::UnifiedMLP;
using ::unified_ml::XGBoostSpec;
using ::unified_ml::distill_to_sle;
using ::unified_ml::explain;
using ::unified_ml::capabilities_of;
using ::unified_ml::capability_descriptor;
using ::unified_ml::capability_descriptor_v;
using ::unified_ml::evaluate;
using ::unified_ml::evaluate_classification;
using ::unified_ml::evaluate_regression;
using sle_backend::BooleanCascade;
using sle_backend::TensorBoundCompilationCache;
using sle_backend::distill_to_logic;
} // namespace unified_ml::stable
