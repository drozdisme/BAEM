#pragma once

/**
 * @file engine_policy.hpp
 * @brief Runtime policy and configuration layer for internal UCAO-backed engine selection.
 */

#include "engine_registry.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace ucao::engine {

enum class EnginePreference {
    RegistryDefault,
    PreferEnabled,
    PreferDisabled,
    ForceEnabled,
    ForceDisabled,
};

struct FamilyPolicy {
    ModelFamily family = ModelFamily::Unknown;
    EnginePreference preference = EnginePreference::RegistryDefault;
};

struct RuntimePolicy {
    bool ucao_globally_enabled = true;
    std::array<FamilyPolicy, kRegistry.size()> family_policies{};
};

struct SelectionResult {
    ModelFamily family = ModelFamily::Unknown;
    EngineDescriptor descriptor{};
    EnginePreference preference = EnginePreference::RegistryDefault;
    bool selected = false;
};

constexpr RuntimePolicy default_policy() noexcept {
    RuntimePolicy policy{};
    for (std::size_t i = 0; i < kRegistry.size(); ++i) {
        policy.family_policies[i] = FamilyPolicy{kRegistry[i].family, EnginePreference::RegistryDefault};
    }
    return policy;
}

inline RuntimePolicy& runtime_policy_storage() noexcept {
    static RuntimePolicy policy = default_policy();
    return policy;
}

inline RuntimePolicy current_policy() noexcept {
    return runtime_policy_storage();
}

inline void set_policy(const RuntimePolicy& policy) noexcept {
    runtime_policy_storage() = policy;
}

inline void reset_policy() noexcept {
    runtime_policy_storage() = default_policy();
}

inline void set_global_enabled(bool enabled) noexcept {
    runtime_policy_storage().ucao_globally_enabled = enabled;
}

inline void set_family_preference(ModelFamily family, EnginePreference preference) noexcept {
    auto& policy = runtime_policy_storage();
    for (auto& entry : policy.family_policies) {
        if (entry.family == family) {
            entry.preference = preference;
            return;
        }
    }
}

constexpr EnginePreference family_preference(const RuntimePolicy& policy, ModelFamily family) noexcept {
    for (const auto& entry : policy.family_policies) {
        if (entry.family == family) {
            return entry.preference;
        }
    }
    return EnginePreference::RegistryDefault;
}

constexpr bool resolve_enabled(const RuntimePolicy& policy, ModelFamily family, bool registry_enabled) noexcept {
    if (!policy.ucao_globally_enabled) {
        return false;
    }
    switch (family_preference(policy, family)) {
        case EnginePreference::RegistryDefault:
        case EnginePreference::PreferEnabled:
            return registry_enabled;
        case EnginePreference::PreferDisabled:
        case EnginePreference::ForceDisabled:
            return false;
        case EnginePreference::ForceEnabled:
            return true;
    }
    return registry_enabled;
}

constexpr bool sle_engine_supported_family(ModelFamily family) noexcept {
    return family == ModelFamily::Mlp || family == ModelFamily::Pinn || family == ModelFamily::SleEngine;
}

constexpr SelectionResult select_with_policy(const RuntimePolicy& policy, ModelFamily family) noexcept {
    auto descriptor = select(family);
    const auto preference = family_preference(policy, family);
    const bool enabled = resolve_enabled(policy, family, descriptor.enabled);
    if (descriptor.kind == EngineKind::SleEngine && !sle_engine_supported_family(family)) {
        descriptor.enabled = false;
        return SelectionResult{family, descriptor, preference, false};
    }
    descriptor.enabled = enabled;
    return SelectionResult{family, descriptor, preference, enabled};
}

inline SelectionResult select_runtime(ModelFamily family) noexcept {
    return select_with_policy(current_policy(), family);
}

template <ModelFamily Family>
struct PolicyBound : public RegistryBound<Family> {
    static SelectionResult runtime_selection() noexcept { return select_runtime(Family); }
    static bool runtime_uses_ucao_engine() noexcept { return runtime_selection().selected; }
    static EnginePreference runtime_preference() noexcept { return runtime_selection().preference; }
};

inline constexpr const char* to_string(EnginePreference preference) noexcept {
    switch (preference) {
        case EnginePreference::RegistryDefault: return "registry_default";
        case EnginePreference::PreferEnabled: return "prefer_enabled";
        case EnginePreference::PreferDisabled: return "prefer_disabled";
        case EnginePreference::ForceEnabled: return "force_enabled";
        case EnginePreference::ForceDisabled: return "force_disabled";
    }
    return "unknown";
}

} // namespace ucao::engine
