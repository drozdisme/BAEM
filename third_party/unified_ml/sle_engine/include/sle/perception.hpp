#pragma once

#include "sle/bit_vector.hpp"
#include "sle/encoding.hpp"
#include "sle/framework.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sle {

enum class BitstreamSemantic {
    AudioAmplitude,
    VideoIntensity,
    TelemetryScalar,
    Generic,
};

struct BitstreamFrame {
    std::vector<BitVector> channels;
    BitVector target;
    BitstreamSemantic semantic = BitstreamSemantic::Generic;
};

struct ScalarBitstreamConfig {
    EncodingMode encoding_mode = EncodingMode::LfsrStochastic;
    std::size_t stream_length = 256;
    std::uint32_t seed = 0x51EAD123u;
};

struct ChannelStatistics {
    double one_ratio = 0.0;
    double entropy = 0.0;
    double target_correlation = 0.0;
};

struct PerceptionSummary {
    std::vector<ChannelStatistics> channels;
    double target_entropy = 0.0;
    double mean_abs_correlation = 0.0;
};

[[nodiscard]] BitVector encode_probability_bitstream(double value,
                                                     const ScalarBitstreamConfig& config);

[[nodiscard]] Dataset make_bitstream_dataset(const std::vector<std::vector<double>>& feature_rows,
                                             const std::vector<double>& targets,
                                             const ScalarBitstreamConfig& config,
                                             BitstreamSemantic semantic = BitstreamSemantic::Generic);

[[nodiscard]] Dataset make_bitstream_dataset(const std::vector<BitstreamFrame>& frames);

[[nodiscard]] PerceptionSummary summarize_bitstreams(const std::vector<TrainingExample>& dataset);

[[nodiscard]] double score_topology_prior(const BooleanCascade& cascade,
                                          const PerceptionSummary& summary);

} // namespace sle
