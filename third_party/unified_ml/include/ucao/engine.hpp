#pragma once

/**
 * @file engine.hpp
 * @brief Internal engine-layer abstractions built on top of UCAO.
 *
 * These types let model code detect and refer to UCAO-backed internal engine
 * paths without exposing UCAO through the top-level public SDK umbrella.
 */

#include <cstddef>

namespace ucao::engine {

/** @brief Internal engine families available to model implementations. */
enum class EngineKind {
    None,
    PinnClifford,
    CliffordFeatureLayer,
    DeepOperatorGeometry,
    PhysicsInformedOperator,
    SequenceGeometry,
    SparseGeometryDynamics,
    SleEngine,
};

/** @brief Lightweight descriptor for an internal engine-backed execution path. */
struct EngineDescriptor {
    EngineKind kind = EngineKind::None;
    const char* name = "none";
    bool enabled = false;
};

/** @brief Internal engine descriptor for Clifford PINN-style execution. */
inline constexpr EngineDescriptor pinn_clifford{
    EngineKind::PinnClifford,
    "ucao.pinn.clifford",
    true,
};

/** @brief Internal engine descriptor for autograd-facing Clifford feature layers. */
inline constexpr EngineDescriptor clifford_feature_layer{
    EngineKind::CliffordFeatureLayer,
    "ucao.ml.clifford_layer",
    true,
};

/** @brief Internal engine descriptor for geometry-aware operator learning paths. */
inline constexpr EngineDescriptor deep_operator_geometry{
    EngineKind::DeepOperatorGeometry,
    "ucao.deep_onet.geometry",
    true,
};

/** @brief Internal engine descriptor for physics-informed operator-learning paths. */
inline constexpr EngineDescriptor physics_informed_operator{
    EngineKind::PhysicsInformedOperator,
    "ucao.pideeponet.physics_operator",
    true,
};

/** @brief Internal engine descriptor for sequence/transform geometry paths. */
inline constexpr EngineDescriptor sequence_geometry{
    EngineKind::SequenceGeometry,
    "ucao.transformer.sequence_geometry",
    true,
};

/** @brief Internal engine descriptor for sparse geometry-aware dynamics paths. */
inline constexpr EngineDescriptor sparse_geometry_dynamics{
    EngineKind::SparseGeometryDynamics,
    "ucao.sindy.sparse_geometry",
    true,
};

/** @brief Internal engine descriptor for SLE-backed logic distillation and execution. */
inline constexpr EngineDescriptor sle_engine{
    EngineKind::SleEngine,
    "ucao.sle.engine",
    true,
};

} // namespace ucao::engine
