#pragma once

#include "sle/bit_vector.hpp"
#include "sle/simd.hpp"

#include <array>
#include <cstdint>

namespace sle {

[[nodiscard]] BitVector ternary_apply_fast(const BitVector& a,
                                           const BitVector& b,
                                           const BitVector& c,
                                           std::uint8_t mask);

[[nodiscard]] std::array<std::uint8_t, 256> build_byte_lut(const std::array<std::uint8_t, 8>& rows) noexcept;

[[nodiscard]] BitVector apply_byte_map_fast(const BitVector& input,
                                            const std::array<std::uint8_t, 256>& lut);

} // namespace sle
