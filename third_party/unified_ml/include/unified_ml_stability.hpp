#pragma once

namespace unified_ml::stability {

enum class Tier {
    Core,
    Stable,
    Experimental,
};

constexpr const char* to_string(Tier tier) noexcept {
    switch (tier) {
        case Tier::Core: return "core";
        case Tier::Stable: return "stable";
        case Tier::Experimental: return "experimental";
    }
    return "unknown";
}

} // namespace unified_ml::stability
