#pragma once

#include <span>
#include <stdexcept>
#include <string>

namespace core {

/**
 * @brief Base exception type for unified_ml public API failures.
 */
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Non-owning view over contiguous immutable scalar data.
 */
using ConstRealSpan = std::span<const double>;

/**
 * @brief Non-owning view over contiguous mutable scalar data.
 */
using RealSpan = std::span<double>;

/**
 * @brief Validate that a span is non-empty.
 * @param values Input span.
 * @param name Logical argument name.
 * @throws Error if the span is empty.
 */
inline void require_non_empty(ConstRealSpan values, const std::string& name) {
    if (values.empty()) {
        throw Error(name + " must not be empty");
    }
}

} // namespace core
