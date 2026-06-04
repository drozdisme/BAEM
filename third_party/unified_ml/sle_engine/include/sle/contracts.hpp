#pragma once

#include "sle/bit_vector.hpp"

#include <stdexcept>

namespace sle {

struct HardLogicContract {
    BitVector required;
    BitVector forbidden;
};

[[nodiscard]] inline BitVector apply_forbidden_filter(const BitVector& value, const BitVector& forbidden) {
    if (value.size() != forbidden.size()) throw std::invalid_argument("forbidden filter size mismatch");
    return value.bit_and(forbidden.bit_not());
}

[[nodiscard]] inline bool required_ok(const BitVector& value, const BitVector& required) {
    if (value.size() != required.size()) throw std::invalid_argument("required verification size mismatch");
    return value.bit_and(required).to_string() == required.to_string();
}

[[nodiscard]] inline bool contract_ok(const BitVector& value, const HardLogicContract& contract) {
    return required_ok(value, contract.required) && apply_forbidden_filter(value, contract.forbidden).to_string() == value.to_string();
}

} // namespace sle
