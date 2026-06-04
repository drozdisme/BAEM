#pragma once

#include "sle/engine.hpp"

#include <string>

namespace sle {

struct CascadeValidationResult {
    bool ok = true;
    std::size_t gate_index = 0;
    std::string message;
};

[[nodiscard]] CascadeValidationResult validate_cascade(const BooleanCascade& cascade) noexcept;
[[nodiscard]] bool is_valid_cascade(const BooleanCascade& cascade) noexcept;

} // namespace sle
