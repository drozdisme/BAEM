#include "models/sle/distillation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace core::models::sle {
namespace {

struct PackedBitMatrix {
    std::size_t sample_count = 0;
    std::size_t word_count = 0;
    std::vector<std::vector<std::uint64_t>> columns;
};

[[nodiscard]] double entropy_from_counts(double positives, double negatives) {
    const double total = positives + negatives;
    if (total <= 0.0) return 0.0;
    const auto term = [total](double count) {
        if (count <= 0.0) return 0.0;
        const double p = count / total;
        return -p * std::log2(p);
    };
    return term(positives) + term(negatives);
}

[[nodiscard]] PackedBitMatrix pack_quantized_columns(std::span<const std::vector<double>> features,
                                                     std::span<const double> thresholds) {
    PackedBitMatrix packed;
    packed.sample_count = features.size();
    packed.word_count = (features.size() + 63U) / 64U;
    packed.columns.assign(thresholds.size(), std::vector<std::uint64_t>(packed.word_count, 0));

    for (std::size_t feature_idx = 0; feature_idx < thresholds.size(); ++feature_idx) {
        auto& column = packed.columns[feature_idx];
        const double threshold = thresholds[feature_idx];

#if defined(__AVX512F__)
        constexpr std::size_t lanes = 8;
        std::size_t base = 0;
        alignas(64) double lane_values[lanes];
        while (base + lanes <= features.size()) {
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                lane_values[lane] = features[base + lane][feature_idx];
            }
            const __m512d vals = _mm512_load_pd(lane_values);
            const __m512d thr = _mm512_set1_pd(threshold);
            const __mmask8 mask = _mm512_cmp_pd_mask(vals, thr, _CMP_GE_OQ);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                if ((mask >> lane) & 0x1U) {
                    const std::size_t sample = base + lane;
                    column[sample / 64U] |= (std::uint64_t{1} << (sample % 64U));
                }
            }
            base += lanes;
        }
        for (std::size_t sample = base; sample < features.size(); ++sample) {
            if (features[sample][feature_idx] >= threshold) {
                column[sample / 64U] |= (std::uint64_t{1} << (sample % 64U));
            }
        }
#else
        for (std::size_t sample = 0; sample < features.size(); ++sample) {
            if (features[sample][feature_idx] >= threshold) {
                column[sample / 64U] |= (std::uint64_t{1} << (sample % 64U));
            }
        }
#endif
    }

    return packed;
}

[[nodiscard]] DistillDataset build_dataset_from_packed(const PackedBitMatrix& packed,
                                                       std::span<const double> predictions,
                                                       double target_threshold,
                                                       std::span<const double> thresholds) {
    DistillDataset dataset;
    dataset.thresholds.assign(thresholds.begin(), thresholds.end());
    dataset.feature_count = packed.columns.size();

    std::vector<std::size_t> selected_indices;
    selected_indices.reserve(packed.sample_count);
    const std::size_t max_examples = 64;
    if (packed.sample_count <= max_examples) {
        for (std::size_t i = 0; i < packed.sample_count; ++i) selected_indices.push_back(i);
    } else {
        std::vector<std::size_t> positives;
        std::vector<std::size_t> negatives;
        positives.reserve(packed.sample_count / 2);
        negatives.reserve(packed.sample_count / 2);
        for (std::size_t i = 0; i < packed.sample_count; ++i) {
            if (predictions[i] >= target_threshold) positives.push_back(i);
            else negatives.push_back(i);
        }
        const std::size_t positive_quota = std::min(max_examples / 2, positives.size());
        const std::size_t negative_quota = std::min(max_examples - positive_quota, negatives.size());
        auto take_evenly = [&](const std::vector<std::size_t>& src, std::size_t quota) {
            if (quota == 0 || src.empty()) return;
            if (quota >= src.size()) {
                selected_indices.insert(selected_indices.end(), src.begin(), src.end());
                return;
            }
            const double step = static_cast<double>(src.size() - 1) / static_cast<double>(quota - 1);
            for (std::size_t q = 0; q < quota; ++q) {
                selected_indices.push_back(src[static_cast<std::size_t>(std::round(step * q))]);
            }
        };
        take_evenly(positives, positive_quota);
        take_evenly(negatives, negative_quota);
        if (selected_indices.size() < max_examples) {
            for (std::size_t i = 0; i < packed.sample_count && selected_indices.size() < max_examples; ++i) {
                if (std::find(selected_indices.begin(), selected_indices.end(), i) == selected_indices.end()) {
                    selected_indices.push_back(i);
                }
            }
        }
        std::sort(selected_indices.begin(), selected_indices.end());
        selected_indices.erase(std::unique(selected_indices.begin(), selected_indices.end()), selected_indices.end());
    }

    dataset.examples.reserve(selected_indices.size());
    for (std::size_t sample : selected_indices) {
        std::vector<::sle::BitVector> inputs;
        inputs.reserve(packed.columns.size());
        for (const auto& column : packed.columns) {
            ::sle::BitVector bit(1);
            const bool value = ((column[sample / 64U] >> (sample % 64U)) & 0x1U) != 0;
            bit.set(0, value);
            inputs.push_back(std::move(bit));
        }

        ::sle::BitVector target(1);
        target.set(0, predictions[sample] >= target_threshold);
        dataset.examples.push_back(::sle::TrainingExample{std::move(inputs), std::move(target)});
    }

    return dataset;
}

} // namespace

DistillDatasetBuilder::DistillDatasetBuilder(DistillConfig config) : config_(std::move(config)) {}

double DistillDatasetBuilder::information_gain(std::span<const double> values,
                                               std::span<const double> labels,
                                               double threshold) {
    double left_pos = 0.0, left_neg = 0.0, right_pos = 0.0, right_neg = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const bool positive = labels[i] >= 0.5;
        if (values[i] < threshold) {
            positive ? ++left_pos : ++left_neg;
        } else {
            positive ? ++right_pos : ++right_neg;
        }
    }

    const double total_pos = left_pos + right_pos;
    const double total_neg = left_neg + right_neg;
    const double total = total_pos + total_neg;
    if (total <= 0.0) return 0.0;

    const double base_entropy = entropy_from_counts(total_pos, total_neg);
    const double left_total = left_pos + left_neg;
    const double right_total = right_pos + right_neg;
    const double weighted = (left_total / total) * entropy_from_counts(left_pos, left_neg) +
                            (right_total / total) * entropy_from_counts(right_pos, right_neg);
    return base_entropy - weighted;
}

double DistillDatasetBuilder::find_optimal_threshold(std::span<const double> values,
                                                     std::span<const double> labels) const {
    if (values.empty()) {
        throw std::invalid_argument("DistillDatasetBuilder::find_optimal_threshold: empty feature column");
    }

    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (sorted.size() == 1) return sorted.front();

    double best_threshold = sorted.front();
    double best_gain = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        const double threshold = 0.5 * (sorted[i - 1] + sorted[i]);
        const double gain = information_gain(values, labels, threshold);
        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = threshold;
        }
    }
    return best_threshold;
}

std::vector<std::uint64_t> DistillDatasetBuilder::quantize_feature(std::span<const double> values,
                                                                   double threshold) {
    const std::size_t word_count = (values.size() + 63U) / 64U;
    std::vector<std::uint64_t> words(word_count, 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] >= threshold) {
            words[i / 64U] |= (std::uint64_t{1} << (i % 64U));
        }
    }
    return words;
}

DistillDataset DistillDatasetBuilder::build(std::span<const std::vector<double>> features,
                                            std::span<const double> predictions) const {
    if (features.empty()) throw std::invalid_argument("DistillDatasetBuilder::build: empty feature matrix");
    if (features.size() != predictions.size()) throw std::invalid_argument("DistillDatasetBuilder::build: features/predictions size mismatch");

    const std::size_t feature_count = features.front().size();
    if (feature_count == 0) throw std::invalid_argument("DistillDatasetBuilder::build: zero feature count");
    for (const auto& row : features) {
        if (row.size() != feature_count) throw std::invalid_argument("DistillDatasetBuilder::build: ragged feature matrix");
    }

    std::vector<double> labels(predictions.begin(), predictions.end());
    const double target_threshold = find_optimal_threshold(std::span<const double>(predictions.data(), predictions.size()),
                                                           std::span<const double>(predictions.data(), predictions.size()));

    std::vector<double> thresholds(feature_count, config_.threshold);
    std::vector<double> column(features.size());
    for (std::size_t feature_idx = 0; feature_idx < feature_count; ++feature_idx) {
        for (std::size_t sample = 0; sample < features.size(); ++sample) {
            column[sample] = features[sample][feature_idx];
        }
        thresholds[feature_idx] = find_optimal_threshold(std::span<const double>(column.data(), column.size()),
                                                         std::span<const double>(labels.data(), labels.size()));
    }

    const auto packed = pack_quantized_columns(features, thresholds);
    return build_dataset_from_packed(packed, predictions, target_threshold, thresholds);
}

BooleanCascade distill_to_circuit(const DistillDataset& dataset, const DistillConfig& config) {
    if (dataset.examples.empty()) {
        throw std::invalid_argument("distill_to_circuit: dataset is empty");
    }
    const std::size_t input_count = dataset.examples.front().inputs.size();
    if (input_count == 0) {
        throw std::invalid_argument("distill_to_circuit: dataset must contain at least one input bit-vector");
    }

    ::sle::FullEngineConfig engine_config{};
    engine_config.gate_count = config.gate_budget;
    engine_config.hlc_aware = true;
    engine_config.synthesis = config.synthesis;
    engine_config.synthesis.iterations = std::max<std::size_t>(engine_config.synthesis.iterations, config.gate_budget * 64);
    engine_config.solver_policy.tiers = {
        {::sle::SynthesisTier::Exact, true, 0, true},
        {::sle::SynthesisTier::Local, true, 0, true},
        {::sle::SynthesisTier::MonteCarloTreeSearch, false, 0, false},
    };
    engine_config.solver_policy.residual_policy = ::sle::ResidualPolicyMode::Disabled;
    engine_config.solver_policy.enable_topology_mutation = false;
    engine_config.residual_planner.tiers = {
        {::sle::SynthesisTier::Exact, false, 0, false},
        {::sle::SynthesisTier::Local, false, 0, false},
        {::sle::SynthesisTier::MonteCarloTreeSearch, false, 0, false},
    };

    auto trained = ::sle::train_full_engine(dataset.examples, engine_config, {});
    const auto& base = trained.model.base;
    if (base.gate_count() > config.gate_budget) {
        throw std::runtime_error("distill_to_circuit: synthesized circuit exceeded gate budget");
    }
    return base;
}

BooleanCascade distill_to_circuit(const rf::RandomForest& model, const DistillDataset& dataset, const DistillConfig& config) {
    (void)model;
    return distill_to_circuit(dataset, config);
}

BooleanCascade distill_to_circuit(const xgb::XGBModel& model, const DistillDataset& dataset, const DistillConfig& config) {
    (void)model;
    return distill_to_circuit(dataset, config);
}

BooleanCascade distill_to_circuit(const iforest::IsolationForest& model, const DistillDataset& dataset, const DistillConfig& config) {
    (void)model;
    return distill_to_circuit(dataset, config);
}

BooleanCascade distill_to_circuit(const pca::PCA& model, const DistillDataset& dataset, const DistillConfig& config) {
    (void)model;
    return distill_to_circuit(dataset, config);
}

} // namespace core::models::sle
