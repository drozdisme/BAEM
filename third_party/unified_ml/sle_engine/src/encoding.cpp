#include "sle/encoding.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sle {

BitVector encode_binary_direct(const std::vector<bool>& values) {
    BitVector out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out.set(i, values[i]);
    return out;
}

BitVector encode_delta_sigma(double x, std::size_t length, std::uint32_t scale) {
    if (x < 0.0 || x > 1.0) throw std::invalid_argument("delta-sigma input must be in [0,1]");
    BitVector out(length);
    std::int64_t acc = 0;
    const auto step = static_cast<std::int64_t>(std::llround(x * static_cast<double>(scale)));
    const auto threshold = static_cast<std::int64_t>(scale / 2u);
    for (std::size_t t = 0; t < length; ++t) {
        acc += step;
        const bool bit = acc > threshold;
        out.set(t, bit);
        if (bit) acc -= static_cast<std::int64_t>(scale);
    }
    return out;
}

BitVector encode_lfsr_stochastic(double p, std::size_t length, std::uint32_t seed) {
    if (p < 0.0 || p > 1.0) throw std::invalid_argument("lfsr input must be in [0,1]");
    BitVector out(length);
    std::uint32_t state = seed ? seed : 0xACE1u;
    const auto threshold = static_cast<std::uint32_t>(p * static_cast<double>(0xFFFFFFFFu));
    for (std::size_t i = 0; i < length; ++i) {
        const std::uint32_t lsb = state & 1u;
        state >>= 1u;
        if (lsb != 0u) state ^= 0x80200003u;
        out.set(i, state <= threshold);
    }
    return out;
}

EncodedBatch encode_real_features(const std::vector<double>& values,
                                  EncodingMode mode,
                                  std::size_t stream_length) {
    EncodedBatch batch;
    batch.mode = mode;
    batch.streams.reserve(values.size());
    for (double value : values) {
        switch (mode) {
            case EncodingMode::BinaryDirect: {
                // Encode value as a thermometer / unary code of length stream_length:
                // the first floor(value * stream_length) bits are 1, the rest are 0.
                // This gives a stream whose density == value, but in a deterministic
                // packed form (LSB-aligned ones) that is compatible with the stochastic
                // logic framework while providing a correct bit-density encoding.
                const std::size_t L = stream_length > 0 ? stream_length : 1;
                BitVector v(L);
                const std::size_t ones = static_cast<std::size_t>(
                    std::clamp(std::round(value * static_cast<double>(L)),
                               0.0, static_cast<double>(L)));
                for (std::size_t i = 0; i < ones; ++i) v.set(i, true);
                batch.streams.push_back(v);
                break;
            }
            case EncodingMode::DeltaSigma:
                batch.streams.push_back(encode_delta_sigma(value, stream_length));
                break;
            case EncodingMode::LfsrStochastic:
                batch.streams.push_back(encode_lfsr_stochastic(value, stream_length));
                break;
        }
    }
    return batch;
}

} // namespace sle