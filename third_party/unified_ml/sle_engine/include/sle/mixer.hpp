#pragma once

#include "sle/bit_vector.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sle {

class GF2ByteMatrix {
public:
    GF2ByteMatrix();

    [[nodiscard]] static GF2ByteMatrix identity();
    [[nodiscard]] static GF2ByteMatrix from_row_xors(const std::vector<std::pair<std::uint8_t, std::uint8_t>>& ops);

    [[nodiscard]] std::uint8_t apply(std::uint8_t value) const noexcept;
    [[nodiscard]] GF2ByteMatrix inverse() const;
    [[nodiscard]] bool is_invertible() const noexcept;
    [[nodiscard]] const std::array<std::uint8_t, 8>& rows() const noexcept { return rows_; }

private:
    std::array<std::uint8_t, 8> rows_{};
};

class BitMixer {
public:
    explicit BitMixer(GF2ByteMatrix matrix = GF2ByteMatrix::identity());

    [[nodiscard]] BitVector mix(const BitVector& input) const;
    [[nodiscard]] BitVector unmix(const BitVector& input) const;

private:
    GF2ByteMatrix matrix_;
    std::array<std::uint8_t, 256> lut_{};
};

} // namespace sle

