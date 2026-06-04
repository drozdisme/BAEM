#include "sle/validation.hpp"

namespace sle {

CascadeValidationResult validate_cascade(const BooleanCascade& cascade) noexcept {
    if (cascade.input_count() == 0) {
        return {false, 0, "cascade must contain at least one input"};
    }

    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        const auto& gate = cascade.gates()[gate_index];
        const std::size_t available = cascade.input_count() + gate_index;
        if (gate.a >= available || gate.b >= available || gate.c >= available) {
            return {false, gate_index, "gate references future or missing node"};
        }
    }

    return {};
}

bool is_valid_cascade(const BooleanCascade& cascade) noexcept {
    return validate_cascade(cascade).ok;
}

} // namespace sle
