#include "sle/perception.hpp"

#include "sle/synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sle {
namespace {

double safe_entropy(double p) {
    const double clamped = std::clamp(p, 1e-9, 1.0 - 1e-9);
    return -(clamped * std::log2(clamped) + (1.0 - clamped) * std::log2(1.0 - clamped));
}

double bit_correlation(const BitVector& lhs, const BitVector& rhs) {
    if (lhs.size() != rhs.size() || lhs.size() == 0) return 0.0;
    double mean_l = 0.0;
    double mean_r = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        mean_l += lhs.get(i) ? 1.0 : 0.0;
        mean_r += rhs.get(i) ? 1.0 : 0.0;
    }
    mean_l /= static_cast<double>(lhs.size());
    mean_r /= static_cast<double>(rhs.size());

    double cov = 0.0;
    double var_l = 0.0;
    double var_r = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const double lv = (lhs.get(i) ? 1.0 : 0.0) - mean_l;
        const double rv = (rhs.get(i) ? 1.0 : 0.0) - mean_r;
        cov += lv * rv;
        var_l += lv * lv;
        var_r += rv * rv;
    }
    if (var_l <= 1e-12 || var_r <= 1e-12) return 0.0;
    return cov / std::sqrt(var_l * var_r);
}

} // namespace

BitVector encode_probability_bitstream(double value, const ScalarBitstreamConfig& config) {
    const double clamped = std::clamp(value, 0.0, 1.0);
    return config.encoding_mode == EncodingMode::DeltaSigma
        ? encode_delta_sigma(clamped, config.stream_length)
        : encode_lfsr_stochastic(clamped, config.stream_length, config.seed);
}

Dataset make_bitstream_dataset(const std::vector<std::vector<double>>& feature_rows,
                               const std::vector<double>& targets,
                               const ScalarBitstreamConfig& config,
                               BitstreamSemantic semantic) {
    if (feature_rows.size() != targets.size()) {
        throw std::invalid_argument("feature_rows and targets must have the same size");
    }

    std::vector<Sample> samples;
    samples.reserve(feature_rows.size());

    for (std::size_t row = 0; row < feature_rows.size(); ++row) {
        std::vector<BitVector> channels;
        channels.reserve(feature_rows[row].size());
        for (double value : feature_rows[row]) {
            channels.push_back(encode_probability_bitstream(value, config));
        }
        samples.push_back(Sample{std::move(channels), encode_probability_bitstream(targets[row], config)});
        (void)semantic;
    }

    return Dataset(std::move(samples), TaskKind::StochasticBitstream);
}

Dataset make_bitstream_dataset(const std::vector<BitstreamFrame>& frames) {
    std::vector<Sample> samples;
    samples.reserve(frames.size());
    for (const auto& frame : frames) {
        samples.push_back(Sample{frame.channels, frame.target});
    }
    return Dataset(std::move(samples), TaskKind::StochasticBitstream);
}

PerceptionSummary summarize_bitstreams(const std::vector<TrainingExample>& dataset) {
    PerceptionSummary summary;
    if (dataset.empty()) return summary;

    const std::size_t channel_count = dataset.front().inputs.size();
    summary.channels.resize(channel_count);

    double target_ones = 0.0;
    double target_bits = 0.0;
    for (const auto& ex : dataset) {
        target_ones += static_cast<double>(ex.target.popcount());
        target_bits += static_cast<double>(ex.target.size());
    }
    if (target_bits > 0.0) {
        summary.target_entropy = safe_entropy(target_ones / target_bits);
    }

    for (std::size_t channel = 0; channel < channel_count; ++channel) {
        double ones = 0.0;
        double bits = 0.0;
        double corr = 0.0;
        for (const auto& ex : dataset) {
            ones += static_cast<double>(ex.inputs[channel].popcount());
            bits += static_cast<double>(ex.inputs[channel].size());
            corr += std::abs(bit_correlation(ex.inputs[channel], ex.target));
        }
        auto& stats = summary.channels[channel];
        stats.one_ratio = bits > 0.0 ? ones / bits : 0.0;
        stats.entropy = bits > 0.0 ? safe_entropy(stats.one_ratio) : 0.0;
        stats.target_correlation = corr / static_cast<double>(std::max<std::size_t>(1, dataset.size()));
        summary.mean_abs_correlation += stats.target_correlation;
    }

    summary.mean_abs_correlation /= static_cast<double>(std::max<std::size_t>(1, summary.channels.size()));
    return summary;
}

double score_topology_prior(const BooleanCascade& cascade, const PerceptionSummary& summary) {
    if (summary.channels.empty()) return 0.0;

    double score = 0.0;
    for (const auto& gate : cascade.gates()) {
        const std::size_t refs[3] = {gate.a, gate.b, gate.c};
        for (std::size_t source : refs) {
            if (source < summary.channels.size()) {
                const auto& stats = summary.channels[source];
                score += 0.55 * stats.target_correlation + 0.25 * stats.entropy + 0.20 * (1.0 - std::abs(0.5 - stats.one_ratio) * 2.0);
            } else {
                score += 0.35 * summary.mean_abs_correlation + 0.15 * summary.target_entropy;
            }
        }

        if (gate.mask == 0x96) score += 0.75 + summary.mean_abs_correlation;
        if (gate.mask == 0xCA || gate.mask == 0xE4) score += 0.45 + 0.5 * summary.mean_abs_correlation;
        if (gate.mask == 0xF0 || gate.mask == 0xCC || gate.mask == 0xAA) score += 0.25;
    }

    return score / static_cast<double>(std::max<std::size_t>(1, cascade.gate_count()));
}

} // namespace sle
