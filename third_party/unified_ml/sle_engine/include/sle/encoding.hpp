#pragma once

#include "sle/bit_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sle {

enum class EncodingMode {
    BinaryDirect,
    DeltaSigma,
    LfsrStochastic,
};

struct EncodedBatch {
    std::vector<BitVector> streams;
    EncodingMode mode = EncodingMode::BinaryDirect;
};

[[nodiscard]] BitVector encode_binary_direct(const std::vector<bool>& values);
[[nodiscard]] BitVector encode_delta_sigma(double x, std::size_t length, std::uint32_t scale = 1u << 16);
[[nodiscard]] BitVector encode_lfsr_stochastic(double p, std::size_t length, std::uint32_t seed = 0xA5A5A5A5u);
[[nodiscard]] EncodedBatch encode_real_features(const std::vector<double>& values,
                                                EncodingMode mode,
                                                std::size_t stream_length);

} // namespace sle
