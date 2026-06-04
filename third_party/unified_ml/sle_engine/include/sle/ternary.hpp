#pragma once

#include "sle/bit_vector.hpp"

#include <cstdint>
#include <stdexcept>

namespace sle {

[[nodiscard]] inline bool ternary_truth(std::uint8_t mask, bool a, bool b, bool c) noexcept {
    const auto index = static_cast<std::uint8_t>((static_cast<unsigned>(a) << 2U) |
                                                 (static_cast<unsigned>(b) << 1U) |
                                                 static_cast<unsigned>(c));
    return ((mask >> index) & 0x1U) != 0U;
}

[[nodiscard]] inline BitVector ternary_apply(const BitVector& a,
                                             const BitVector& b,
                                             const BitVector& c,
                                             std::uint8_t mask) {
    if (a.size() != b.size() || a.size() != c.size()) {
        throw std::invalid_argument("ternary_apply size mismatch");
    }

    BitVector out(a.size());
    for (std::size_t i = 0; i < a.word_count(); ++i) {
        const auto aw = a.words()[i];
        const auto bw = b.words()[i];
        const auto cw = c.words()[i];
        std::uint64_t result = 0;
        for (unsigned bit = 0; bit < 64U; ++bit) {
            const bool abit = ((aw >> bit) & 1ULL) != 0ULL;
            const bool bbit = ((bw >> bit) & 1ULL) != 0ULL;
            const bool cbit = ((cw >> bit) & 1ULL) != 0ULL;
            if (ternary_truth(mask, abit, bbit, cbit)) result |= (1ULL << bit);
        }
        out.words()[i] = result;
    }
    return out;
}

} // namespace sle
