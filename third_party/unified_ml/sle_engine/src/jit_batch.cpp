#include "sle/jit_batch.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace sle {

namespace {

double estimate_batch_model_bits(const BooleanCascade& cascade, const SynthesisConfig& config) {
    return estimate_description_length_bits(cascade, config);
}

std::size_t aligned_words(std::size_t bits, std::size_t alignment_bytes) {
    const auto words = (bits + 63U) / 64U;
    const auto alignment_words = std::max<std::size_t>(1, alignment_bytes / sizeof(std::uint64_t));
    return ((words + alignment_words - 1U) / alignment_words) * alignment_words;
}

} // namespace

bool PreparedBatch::is_aligned() const noexcept {
    if (target_ptr == nullptr) return false;
    if ((reinterpret_cast<std::uintptr_t>(target_ptr) % alignment) != 0U) return false;
    for (const auto* ptr : input_ptrs) {
        if ((reinterpret_cast<std::uintptr_t>(ptr) % alignment) != 0U) return false;
    }
    return true;
}

PreparedBatch prepare_batch(const std::vector<TrainingExample>& dataset,
                            std::size_t alignment,
                            std::size_t register_bits) {
    if (dataset.empty()) throw std::invalid_argument("prepare_batch requires non-empty dataset");
    if (alignment == 0 || (alignment % alignof(std::uint64_t)) != 0) throw std::invalid_argument("prepare_batch alignment must be uint64-aligned");
    if (register_bits == 0 || (register_bits % 64U) != 0U) throw std::invalid_argument("prepare_batch register_bits must be multiple of 64");

    const auto input_count = dataset.front().inputs.size();
    const auto feature_bits = dataset.front().target.size();
    for (const auto& ex : dataset) {
        if (ex.inputs.size() != input_count) throw std::invalid_argument("prepare_batch input count mismatch");
        if (ex.target.size() != feature_bits) throw std::invalid_argument("prepare_batch target size mismatch");
        for (const auto& input : ex.inputs) {
            if (input.size() != feature_bits) throw std::invalid_argument("prepare_batch input bit width mismatch");
        }
    }

    const auto lane_examples = register_bits;
    const auto padded_examples = ((dataset.size() + lane_examples - 1U) / lane_examples) * lane_examples;
    const auto lane_words = padded_examples / 64U;
    const auto plane_words = aligned_words(padded_examples, alignment);
    const auto total_planes = input_count + 1U;
    const auto total_words = total_planes * feature_bits * plane_words;

    auto storage = std::shared_ptr<std::uint64_t[]>(new std::uint64_t[total_words](), std::default_delete<std::uint64_t[]>());

    PreparedBatch batch;
    batch.input_count = input_count;
    batch.example_count = dataset.size();
    batch.padded_examples = padded_examples;
    batch.lane_words = lane_words;
    batch.plane_words = plane_words;
    batch.feature_bits = feature_bits;
    batch.alignment = alignment;
    batch.storage = storage;
    batch.input_ptrs.resize(input_count);

    for (std::size_t input_index = 0; input_index < input_count; ++input_index) {
        auto* plane_base = storage.get() + input_index * feature_bits * plane_words;
        batch.input_ptrs[input_index] = plane_base;
        for (std::size_t example_index = 0; example_index < dataset.size(); ++example_index) {
            const auto& bits = dataset[example_index].inputs[input_index];
            for (std::size_t bit = 0; bit < feature_bits; ++bit) {
                if (!bits.get(bit)) continue;
                const auto word_index = example_index / 64U;
                const auto bit_index = example_index % 64U;
                plane_base[bit * plane_words + word_index] |= (1ULL << bit_index);
            }
        }
    }

    auto* target_base = storage.get() + input_count * feature_bits * plane_words;
    batch.target_ptr = target_base;
    for (std::size_t example_index = 0; example_index < dataset.size(); ++example_index) {
        const auto& bits = dataset[example_index].target;
        for (std::size_t bit = 0; bit < feature_bits; ++bit) {
            if (!bits.get(bit)) continue;
            const auto word_index = example_index / 64U;
            const auto bit_index = example_index % 64U;
            target_base[bit * plane_words + word_index] |= (1ULL << bit_index);
        }
    }

    return batch;
}

SafeBatchedJITEvaluator make_safe_batched_jit_evaluator(const std::vector<TrainingExample>& dataset,
                                                        std::size_t alignment,
                                                        std::size_t register_bits) {
    return SafeBatchedJITEvaluator{prepare_batch(dataset, alignment, register_bits)};
}

FitnessBreakdown SafeBatchedJITEvaluator::evaluate(const CompiledBooleanCascade& compiled,
                                                   const BooleanCascade& cascade,
                                                   const SynthesisConfig& config) const {
    if (batch.input_count != cascade.input_count()) throw std::invalid_argument("SafeBatchedJITEvaluator input count mismatch");
    if (!batch.is_aligned()) throw std::invalid_argument("SafeBatchedJITEvaluator requires aligned prepared batch");

    const auto workspace_words = compiled.workspace_word_count();
    std::vector<std::uint64_t> workspace(workspace_words * batch.feature_bits, 0);
    std::size_t error_bits = 0;

    for (std::size_t bit = 0; bit < batch.feature_bits; ++bit) {
        std::vector<const std::uint64_t*> plane_ptrs;
        plane_ptrs.reserve(batch.input_count);
        for (std::size_t input_index = 0; input_index < batch.input_count; ++input_index) {
            plane_ptrs.push_back(batch.input_ptrs[input_index] + bit * batch.plane_words);
        }
        auto* workspace_base = workspace.data() + bit * workspace_words;
        compiled.run_words(plane_ptrs.data(), workspace_base, batch.lane_words);

        const auto output_node = cascade.input_count() + cascade.gate_count() - 1U;
        const auto* output_words = workspace_base + output_node * batch.lane_words;
        const auto* target_words = batch.target_ptr + bit * batch.plane_words;
        for (std::size_t word = 0; word < batch.lane_words; ++word) {
            std::uint64_t diff = output_words[word] ^ target_words[word];
            if (word == batch.lane_words - 1U && (batch.example_count % 64U) != 0U) {
                const auto live_bits = static_cast<unsigned>(batch.example_count % 64U);
                diff &= ((1ULL << live_bits) - 1ULL);
            }
            error_bits += static_cast<std::size_t>(std::popcount(diff));
        }
    }

    const auto mdl_model_bits = estimate_batch_model_bits(cascade, config);
    double critical_path_cost = 0.0;
    const double computational_cost = estimate_computational_cost(cascade, config, &critical_path_cost);
    const double denom = static_cast<double>(batch.example_count * batch.feature_bits);
    const double output_fitness = 1.0 - static_cast<double>(error_bits) / denom;
    const double mdl_residual_bits = static_cast<double>(error_bits) * config.mdl_error_bit_cost;
    const double mdl_term = config.use_mdl
        ? (mdl_model_bits + mdl_residual_bits) / denom
        : config.complexity_weight * (mdl_model_bits / std::max(1.0, config.mdl_gate_bits));
    const double compute_term = config.computational_cost_weight * (computational_cost / denom)
        + config.critical_path_weight * critical_path_cost;
    const double complexity_penalty = mdl_term + compute_term;
    return FitnessBreakdown{output_fitness, complexity_penalty, mdl_model_bits, mdl_residual_bits, computational_cost, critical_path_cost, output_fitness - complexity_penalty};
}

} // namespace sle
