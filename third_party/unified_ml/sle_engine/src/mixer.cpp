#include "sle/mixer.hpp"

#include "sle/fast_ops.hpp"

#include <algorithm>

namespace sle {

namespace {

std::array<std::uint8_t, 8> make_identity_rows() {
    std::array<std::uint8_t, 8> rows{};
    for (std::uint8_t i = 0; i < 8; ++i) rows[i] = static_cast<std::uint8_t>(1U << i);
    return rows;
}

std::uint8_t multiply_rows(const std::array<std::uint8_t, 8>& rows, std::uint8_t value) noexcept {
    std::uint8_t out = 0;
    for (std::uint8_t i = 0; i < 8; ++i) {
        const auto bits = static_cast<std::uint8_t>(rows[i] & value);
        const auto parity = static_cast<std::uint8_t>(std::popcount(bits) & 0x1);
        out |= static_cast<std::uint8_t>(parity << i);
    }
    return out;
}

} // namespace

GF2ByteMatrix::GF2ByteMatrix() : rows_(make_identity_rows()) {}

GF2ByteMatrix GF2ByteMatrix::identity() {
    return GF2ByteMatrix{};
}

GF2ByteMatrix GF2ByteMatrix::from_row_xors(const std::vector<std::pair<std::uint8_t, std::uint8_t>>& ops) {
    auto matrix = identity();
    for (const auto& [dst, src] : ops) {
        if (dst >= 8 || src >= 8 || dst == src) throw std::invalid_argument("invalid GF(2) row xor operation");
        matrix.rows_[dst] ^= matrix.rows_[src];
    }
    return matrix;
}

std::uint8_t GF2ByteMatrix::apply(std::uint8_t value) const noexcept {
    return multiply_rows(rows_, value);
}

bool GF2ByteMatrix::is_invertible() const noexcept {
    auto rows = rows_;
    std::uint8_t pivot_row = 0;
    for (std::uint8_t col = 0; col < 8; ++col) {
        std::uint8_t selected = 8;
        for (std::uint8_t r = pivot_row; r < 8; ++r) {
            if (((rows[r] >> col) & 0x1U) != 0U) {
                selected = r;
                break;
            }
        }
        if (selected == 8) continue;
        std::swap(rows[pivot_row], rows[selected]);
        for (std::uint8_t r = 0; r < 8; ++r) {
            if (r != pivot_row && (((rows[r] >> col) & 0x1U) != 0U)) rows[r] ^= rows[pivot_row];
        }
        ++pivot_row;
    }
    return pivot_row == 8;
}

GF2ByteMatrix GF2ByteMatrix::inverse() const {
    auto left = rows_;
    auto right = make_identity_rows();

    for (std::uint8_t col = 0; col < 8; ++col) {
        std::uint8_t pivot = col;
        while (pivot < 8 && (((left[pivot] >> col) & 0x1U) == 0U)) ++pivot;
        if (pivot == 8) throw std::runtime_error("GF2ByteMatrix is not invertible");
        if (pivot != col) {
            std::swap(left[col], left[pivot]);
            std::swap(right[col], right[pivot]);
        }
        for (std::uint8_t row = 0; row < 8; ++row) {
            if (row != col && (((left[row] >> col) & 0x1U) != 0U)) {
                left[row] ^= left[col];
                right[row] ^= right[col];
            }
        }
    }

    GF2ByteMatrix out;
    out.rows_ = right;
    return out;
}

BitMixer::BitMixer(GF2ByteMatrix matrix) : matrix_(std::move(matrix)), lut_(build_byte_lut(matrix_.rows())) {}

BitVector BitMixer::mix(const BitVector& input) const {
    return apply_byte_map_fast(input, lut_);
}

BitVector BitMixer::unmix(const BitVector& input) const {
    BitMixer inverse_mixer(matrix_.inverse());
    return inverse_mixer.mix(input);
}

} // namespace sle
