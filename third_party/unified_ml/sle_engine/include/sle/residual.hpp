#pragma once

#include "sle/bit_vector.hpp"

namespace sle {

[[nodiscard]] inline BitVector residual_target(const BitVector& target, const BitVector& base) {
    return target.bit_xor(base);
}

[[nodiscard]] inline BitVector residual_apply(const BitVector& base, const BitVector& residue) {
    return base.bit_xor(residue);
}

} // namespace sle
