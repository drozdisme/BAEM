#include "sle/jit_trainer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace sle {
namespace {

constexpr std::array<std::uint8_t, 6> kPrimitiveMasks{0x00, 0x80, 0x88, 0xE8, 0xFE, 0x96};

std::vector<std::uint8_t> make_expansion_masks(std::uint8_t base_mask,
                                               const BitVector& error_gradient) {
    std::vector<std::uint8_t> masks;
    masks.reserve(kPrimitiveMasks.size() + 8U);
    for (auto mask : kPrimitiveMasks) masks.push_back(mask);
    const auto hint = static_cast<unsigned>(error_gradient.popcount() & 0x7U);
    for (unsigned bit = 0; bit < 8U; ++bit) {
        masks.push_back(static_cast<std::uint8_t>(base_mask ^ (1U << ((bit + hint) & 0x7U))));
    }
    std::sort(masks.begin(), masks.end());
    masks.erase(std::unique(masks.begin(), masks.end()), masks.end());
    return masks;
}

BitVector estimate_error_gradient(const SafeBatchedJITEvaluator& evaluator,
                                  const CompiledBooleanCascade& compiled,
                                  const BooleanCascade& cascade) {
    BitVector gradient(evaluator.batch.feature_bits, false);
    const auto workspace_words = compiled.workspace_word_count();
    std::vector<std::uint64_t> workspace(workspace_words * evaluator.batch.feature_bits, 0);

    for (std::size_t bit = 0; bit < evaluator.batch.feature_bits; ++bit) {
        std::vector<const std::uint64_t*> plane_ptrs;
        plane_ptrs.reserve(evaluator.batch.input_count);
        for (std::size_t input_index = 0; input_index < evaluator.batch.input_count; ++input_index) {
            plane_ptrs.push_back(evaluator.batch.input_ptrs[input_index] + bit * evaluator.batch.plane_words);
        }
        auto* workspace_base = workspace.data() + bit * workspace_words;
        compiled.run_words(plane_ptrs.data(), workspace_base, evaluator.batch.lane_words);
        const auto output_node = cascade.input_count() + cascade.gate_count() - 1U;
        const auto* output_words = workspace_base + output_node * evaluator.batch.lane_words;
        const auto* target_words = evaluator.batch.target_ptr + bit * evaluator.batch.plane_words;
        std::size_t plane_errors = 0;
        for (std::size_t word = 0; word < evaluator.batch.lane_words; ++word) {
            std::uint64_t diff = output_words[word] ^ target_words[word];
            if (word == evaluator.batch.lane_words - 1U && (evaluator.batch.example_count % 64U) != 0U) {
                const auto live_bits = static_cast<unsigned>(evaluator.batch.example_count % 64U);
                diff &= ((1ULL << live_bits) - 1ULL);
            }
            plane_errors += static_cast<std::size_t>(std::popcount(diff));
        }
        if (plane_errors != 0) gradient.set(bit, true);
    }
    return gradient;
}

std::vector<double> score_gates(const BooleanCascade& cascade, const BitVector& error_gradient) {
    std::vector<double> scores(cascade.gate_count(), 0.0);
    const auto pressure = static_cast<double>(error_gradient.popcount() + 1U);
    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        const auto& gate = cascade.gates()[gate_index];
        scores[gate_index] = pressure / static_cast<double>(1U + gate.a + gate.b + gate.c);
    }
    return scores;
}

void synchronize_patch_site(CompiledBooleanCascade& compiled, std::size_t gate_index) {
    if (compiled.patch_points().empty()) return;
    const auto offset = compiled.patch_points()[gate_index].immediate_offset;
    _mm_clflush(compiled.raw_code() + offset);
    _mm_mfence();
    __builtin___clear_cache(reinterpret_cast<char*>(compiled.raw_code() + offset),
                            reinterpret_cast<char*>(compiled.raw_code() + offset + 1U));
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("cpuid" : : "a"(0) : "rbx", "rcx", "rdx", "memory");
#endif
}

} // namespace

JITTrainer::JITTrainer(BooleanCascade initial,
                       const SafeBatchedJITEvaluator& evaluator,
                       SynthesisConfig config,
                       JitCompileOptions jit_options)
    : evaluator_(evaluator), config_(config), jit_options_(jit_options) {
    if (initial.gate_count() == 0) throw std::invalid_argument("JITTrainer requires at least one gate");
    if (jit_options_.preferred_backend == JitBackend::ScalarGpr) {
        throw std::invalid_argument("JITTrainer hot-loop requires patchable backend");
    }
    if (jit_options_.word_count == 0) jit_options_.word_count = evaluator_.batch.lane_words;

    best_.cascade = initial;
    best_.active_bin = compile_boolean_cascade(best_.cascade, jit_options_);
    if (best_.active_bin.patch_points().size() != best_.cascade.gate_count()) {
        throw std::invalid_argument("JITTrainer requires fully patchable compiled cascade");
    }
    best_.fitness = evaluator_.evaluate(best_.active_bin, best_.cascade, config_);

    candidate_.cascade = best_.cascade;
    candidate_.scratch_bin = compile_boolean_cascade(candidate_.cascade, jit_options_);
    if (candidate_.scratch_bin.patch_points().size() != candidate_.cascade.gate_count()) {
        throw std::invalid_argument("JITTrainer requires fully patchable scratch cascade");
    }
    std::memcpy(candidate_.scratch_bin.raw_code(), best_.active_bin.raw_code(), best_.active_bin.code_size());
}

SelectionSignal JITTrainer::compute_selection_signal() const {
    SelectionSignal signal;
    signal.error_gradient = estimate_error_gradient(evaluator_, best_.active_bin, best_.cascade);
    signal.gate_scores = score_gates(best_.cascade, signal.error_gradient);
    return signal;
}

std::vector<std::uint8_t> JITTrainer::expand_masks(std::size_t gate_index,
                                                   const BitVector& error_gradient) const {
    if (gate_index >= best_.cascade.gate_count()) throw std::out_of_range("expand_masks gate index");
    return make_expansion_masks(best_.cascade.gates()[gate_index].mask, error_gradient);
}

void JITTrainer::apply_mutation(std::size_t gate_index, std::uint8_t mask) {
    if (gate_index >= candidate_.cascade.gate_count()) throw std::out_of_range("apply_mutation gate index");
    candidate_.gate_index = gate_index;
    candidate_.previous_mask = candidate_.cascade.gates()[gate_index].mask;
    candidate_.current_mask = mask;
    candidate_.cascade.mutable_gates()[gate_index].mask = mask;
    candidate_.scratch_bin.patch_gate_mask(gate_index, mask, false);
    candidate_active_ = true;
}

void JITTrainer::rollback_mutation() {
    if (!candidate_active_) return;
    candidate_.cascade.mutable_gates()[candidate_.gate_index].mask = candidate_.previous_mask;
    candidate_.scratch_bin.patch_gate_mask(candidate_.gate_index, candidate_.previous_mask, false);
    candidate_active_ = false;
}

bool JITTrainer::evaluate_and_maybe_commit() {
    if (!candidate_active_) throw std::logic_error("evaluate_and_maybe_commit without candidate");
    const auto candidate_fitness = evaluator_.evaluate(candidate_.scratch_bin, candidate_.cascade, config_);
    if (candidate_fitness.total > best_.fitness.total) {
        best_.fitness = candidate_fitness;
        commit_state();
        candidate_active_ = false;
        return true;
    }
    rollback_mutation();
    return false;
}

void JITTrainer::commit_state() {
    best_.cascade = candidate_.cascade;
    best_.active_bin.patch_gate_mask(candidate_.gate_index, candidate_.current_mask, false);
    synchronize_patch_site(best_.active_bin, candidate_.gate_index);
    candidate_.scratch_bin.patch_gate_mask(candidate_.gate_index, candidate_.current_mask, false);
    candidate_.cascade = best_.cascade;
}

SynthesisResult JITTrainer::run(std::size_t iterations) {
    std::size_t accepted = 0;
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        const auto signal = compute_selection_signal();
        const auto gate_index = static_cast<std::size_t>(std::distance(signal.gate_scores.begin(), std::max_element(signal.gate_scores.begin(), signal.gate_scores.end())));
        const auto masks = expand_masks(gate_index, signal.error_gradient);
        for (auto mask : masks) {
            apply_mutation(gate_index, mask);
            if (evaluate_and_maybe_commit()) {
                ++accepted;
                break;
            }
        }
    }
    return SynthesisResult{best_.cascade, best_.fitness, accepted};
}

} // namespace sle
