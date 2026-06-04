#pragma once

#include "sle/jit_batch.hpp"
#include "sle/synthesis.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sle {

struct BestState {
    BooleanCascade cascade;
    CompiledBooleanCascade active_bin;
    FitnessBreakdown fitness;
};

struct CandidateState {
    BooleanCascade cascade;
    CompiledBooleanCascade scratch_bin;
    std::size_t gate_index = 0;
    std::uint8_t previous_mask = 0;
    std::uint8_t current_mask = 0;
};

struct SelectionSignal {
    BitVector error_gradient;
    std::vector<double> gate_scores;
};

class JITTrainer {
public:
    JITTrainer(BooleanCascade initial,
               const SafeBatchedJITEvaluator& evaluator,
               SynthesisConfig config,
               JitCompileOptions jit_options = {});

    [[nodiscard]] const BestState& best_state() const noexcept { return best_; }
    [[nodiscard]] SelectionSignal compute_selection_signal() const;
    [[nodiscard]] std::vector<std::uint8_t> expand_masks(std::size_t gate_index,
                                                         const BitVector& error_gradient) const;
    void apply_mutation(std::size_t gate_index, std::uint8_t mask);
    void rollback_mutation();
    bool evaluate_and_maybe_commit();
    [[nodiscard]] SynthesisResult run(std::size_t iterations);

private:
    void commit_state();

    SafeBatchedJITEvaluator evaluator_;
    SynthesisConfig config_;
    JitCompileOptions jit_options_;
    BestState best_;
    CandidateState candidate_;
    bool candidate_active_ = false;
};

} // namespace sle
