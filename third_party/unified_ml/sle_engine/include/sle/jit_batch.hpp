#pragma once

#include "sle/jit.hpp"
#include "sle/synthesis.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sle {

struct PreparedBatch {
    std::size_t input_count = 0;
    std::size_t example_count = 0;
    std::size_t padded_examples = 0;
    std::size_t lane_words = 0;
    std::size_t plane_words = 0;
    std::size_t feature_bits = 0;
    std::size_t alignment = 64;
    std::vector<const std::uint64_t*> input_ptrs;
    const std::uint64_t* target_ptr = nullptr;
    std::shared_ptr<std::uint64_t[]> storage;

    [[nodiscard]] bool is_aligned() const noexcept;
};

struct SafeBatchedJITEvaluator {
    PreparedBatch batch;

    [[nodiscard]] FitnessBreakdown evaluate(const CompiledBooleanCascade& compiled,
                                            const BooleanCascade& cascade,
                                            const SynthesisConfig& config) const;
};

[[nodiscard]] PreparedBatch prepare_batch(const std::vector<TrainingExample>& dataset,
                                          std::size_t alignment = 64,
                                          std::size_t register_bits = 512);

[[nodiscard]] SafeBatchedJITEvaluator make_safe_batched_jit_evaluator(const std::vector<TrainingExample>& dataset,
                                                                      std::size_t alignment = 64,
                                                                      std::size_t register_bits = 512);

} // namespace sle
