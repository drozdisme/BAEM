#pragma once

/**
 * @file engine_registry.hpp
 * @brief Unified internal registry and selection helpers for UCAO-backed model engines.
 */

#include "engine.hpp"

#include <array>
#include <string_view>

namespace ucao::engine {

/** @brief Internal model families that may select an engine-backed execution path. */
enum class ModelFamily {
    Unknown,
    Mlp,
    Pinn,
    DeepONet,
    PIDeepONet,
    Transformer,
    SINDy,
    CliffordLayer,
    SleEngine,
};

/** @brief Registry entry binding a model family to its preferred internal engine. */
struct RegistryEntry {
    ModelFamily family = ModelFamily::Unknown;
    std::string_view family_name = "unknown";
    EngineDescriptor descriptor{};
};

/** @brief Static registry of current UCAO-backed internal engine mappings. */
inline constexpr std::array<RegistryEntry, 8> kRegistry{{
    {ModelFamily::Mlp, "mlp", sle_engine},
    {ModelFamily::Pinn, "pinn", sle_engine},
    {ModelFamily::DeepONet, "deep_onet", deep_operator_geometry},
    {ModelFamily::PIDeepONet, "pideeponet", physics_informed_operator},
    {ModelFamily::Transformer, "transformer", sequence_geometry},
    {ModelFamily::SINDy, "sindy", sparse_geometry_dynamics},
    {ModelFamily::CliffordLayer, "clifford_layer", clifford_feature_layer},
    {ModelFamily::SleEngine, "sle_engine", sle_engine},
}};

/** @brief Return the registered engine descriptor for a model family. */
constexpr EngineDescriptor select(ModelFamily family) noexcept {
    for (const auto& entry : kRegistry) {
        if (entry.family == family) {
            return entry.descriptor;
        }
    }
    return {};
}

/** @brief Return a stable internal name for a model family. */
constexpr std::string_view model_family_name(ModelFamily family) noexcept {
    for (const auto& entry : kRegistry) {
        if (entry.family == family) {
            return entry.family_name;
        }
    }
    return "unknown";
}

/** @brief Return whether a model family currently has an enabled UCAO-backed path. */
constexpr bool has_enabled_path(ModelFamily family) noexcept {
    return select(family).enabled;
}

/** @brief Generic mixin-style helper for model types with a fixed internal family. */
template <ModelFamily Family>
struct RegistryBound {
    static constexpr ModelFamily model_family() noexcept { return Family; }
    static constexpr bool uses_ucao_engine() noexcept { return has_enabled_path(Family); }
    static constexpr EngineDescriptor engine_descriptor() noexcept { return select(Family); }
    static constexpr std::string_view engine_family_name() noexcept { return model_family_name(Family); }
};

} // namespace ucao::engine
