#pragma once

#include "sle/perception.hpp"
#include "sle/synthesis.hpp"
#include "sle/topology_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sle {

enum class SearchStrategy {
    LocalGreedy,
    MonteCarloTreeSearch,
};

enum class MutationKind {
    FlipMaskBit,
    RewireInputA,
    RewireInputB,
    RewireInputC,
    MacroTemplate,
};

enum class MacroTemplateKind {
    None,
    ThreeInputXor,
    Multiplexer,
    DelayCascade,
    TwoStageXor,
};

struct MutationAction {
    MutationKind kind = MutationKind::FlipMaskBit;
    std::size_t gate_index = 0;
    std::size_t operand_slot = 0;
    std::size_t new_source = 0;
    std::uint8_t bit_index = 0;
    MacroTemplateKind macro_kind = MacroTemplateKind::None;
    std::size_t macro_span = 0;
};

struct SearchTraceEntry {
    std::size_t depth = 0;
    std::size_t visits = 0;
    double value = 0.0;
    FitnessBreakdown fitness;
    MutationAction action;
};

struct SearchDiagnostics {
    SearchStrategy strategy = SearchStrategy::LocalGreedy;
    std::size_t explored_nodes = 0;
    std::size_t expanded_nodes = 0;
    std::size_t rollout_evaluations = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::vector<SearchTraceEntry> principal_path;
};

[[nodiscard]] std::vector<MutationAction> enumerate_mutation_actions(const BooleanCascade& cascade);

[[nodiscard]] std::vector<MutationAction> enumerate_macro_actions(const BooleanCascade& cascade);

[[nodiscard]] BooleanCascade apply_mutation_action(const BooleanCascade& cascade,
                                                   const MutationAction& action);

[[nodiscard]] std::string describe_mutation_action(const MutationAction& action);

[[nodiscard]] SynthesisResult synthesize_mcts(BooleanCascade initial,
                                              const std::vector<TrainingExample>& dataset,
                                              const SynthesisConfig& config,
                                              SearchDiagnostics* diagnostics = nullptr,
                                              const PerceptionSummary* perception = nullptr,
                                              CompilationCache* cache = nullptr);

[[nodiscard]] SynthesisResult synthesize_local_greedy(BooleanCascade initial,
                                                      const std::vector<TrainingExample>& dataset,
                                                      const SynthesisConfig& config,
                                                      SearchDiagnostics* diagnostics = nullptr,
                                                      const PerceptionSummary* perception = nullptr,
                                                      CompilationCache* cache = nullptr);

} // namespace sle
