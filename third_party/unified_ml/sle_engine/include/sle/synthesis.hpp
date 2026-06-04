#pragma once

#include "sle/engine.hpp"
#include "sle/jit.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace sle {

enum class TaskKind : int;

struct TrainingExample {
    std::vector<BitVector> inputs;
    BitVector target;
};

struct FitnessBreakdown {
    double output_fitness = 0.0;
    double complexity_penalty = 0.0;
    double mdl_model_bits = 0.0;
    double mdl_residual_bits = 0.0;
    double computational_cost = 0.0;
    double critical_path_cost = 0.0;
    double total = 0.0;
};

struct SynthesisConfig {
    std::size_t iterations = 4096;
    double complexity_weight = 0.0025;
    double mdl_gate_bits = 32.0;
    double mdl_topology_bits = 12.0;
    double mdl_error_bit_cost = 1.0;
    double computational_cost_weight = 0.04;
    double critical_path_weight = 0.025;
    double perception_prior_weight = 0.08;
    std::size_t rollout_budget = 64;
    std::size_t rollout_patience = 12;
    double uct_exploration_base = 1.41421356237;
    double uct_epoch_decay = 0.35;
    double uct_mdl_sensitivity = 0.45;
    bool use_mdl = true;
    bool prefer_jit = true;
    std::uint64_t seed = 0xC0FFEEULL;
    double target_output_fitness = 1.0;
};

struct SynthesisResult {
    BooleanCascade cascade;
    FitnessBreakdown fitness;
    std::size_t accepted_mutations = 0;
};

enum class SynthesisTier {
    Exact,
    Local,
    MonteCarloTreeSearch,
    Residual,
};

enum class ExactPatternKind {
    None,
    TruthTableMask,
    IdentityPassThrough,
    SimpleMux,
    XorChain3,
    XorChainN,
    ParityFamily,
};

enum class ResidualPolicyMode {
    Disabled,
    FallbackOnly,
    AlwaysRefine,
};

struct TierExecutionConfig {
    SynthesisTier tier = SynthesisTier::Local;
    bool enabled = true;
    std::size_t iterations_override = 0;
    bool stop_on_target = true;
};

struct SolverPolicy {
    std::vector<TierExecutionConfig> tiers{
        {SynthesisTier::Exact, true, 0, true},
        {SynthesisTier::Local, true, 0, true},
        {SynthesisTier::MonteCarloTreeSearch, true, 0, true},
    };
    ResidualPolicyMode residual_policy = ResidualPolicyMode::FallbackOnly;
    bool enable_topology_mutation = true;
};

struct ExactPatternMatch {
    ExactPatternKind kind = ExactPatternKind::None;
    BooleanCascade cascade;
    std::string name;
    std::string description;
};

struct TierDiagnostics {
    SynthesisTier tier = SynthesisTier::Local;
    std::string tier_name;
    bool executed = false;
    bool reached_target = false;
    FitnessBreakdown fitness;
    std::size_t accepted_mutations = 0;
    std::string summary;
};

struct ResidualPlanStep {
    std::string name;
    SynthesisResult result;
    FitnessBreakdown final_fitness;
    std::string summary;
};

struct ResidualPlannerConfig {
    std::vector<TierExecutionConfig> tiers{
        {SynthesisTier::Exact, true, 0, false},
        {SynthesisTier::Local, true, 0, false},
        {SynthesisTier::MonteCarloTreeSearch, true, 0, false},
    };
};

struct TieredSynthesisCandidate {
    SynthesisTier tier = SynthesisTier::Local;
    SynthesisResult result;
    TierDiagnostics diagnostics;
};

struct ResidualSynthesisResult {
    SynthesisResult base;
    SynthesisResult residual;
    FitnessBreakdown final_fitness;
    std::string summary;
    std::vector<ResidualPlanStep> steps;
};

struct JitMutationState {
    BooleanCascade cascade;
    CompiledBooleanCascade compiled;
    FitnessBreakdown fitness;
};

[[nodiscard]] double estimate_description_length_bits(const BooleanCascade& cascade,
                                                      const SynthesisConfig& config);

[[nodiscard]] double estimate_computational_cost(const BooleanCascade& cascade,
                                                 const SynthesisConfig& config,
                                                 double* critical_path_cost = nullptr);

[[nodiscard]] FitnessBreakdown evaluate_fitness(const BooleanCascade& cascade,
                                                const std::vector<TrainingExample>& dataset,
                                                const SynthesisConfig& config);

[[nodiscard]] FitnessBreakdown evaluate_fitness(const CompiledBooleanCascade& compiled,
                                                const BooleanCascade& cascade,
                                                const std::vector<TrainingExample>& dataset,
                                                const SynthesisConfig& config);

[[nodiscard]] FitnessBreakdown evaluate_fitness_residual(const BooleanCascade& base,
                                                         const BooleanCascade& residual,
                                                         const std::vector<TrainingExample>& dataset,
                                                         const SynthesisConfig& config);

[[nodiscard]] SynthesisResult synthesize_local(BooleanCascade initial,
                                               const std::vector<TrainingExample>& dataset,
                                               const SynthesisConfig& config);

[[nodiscard]] SynthesisResult synthesize_exact(BooleanCascade initial,
                                               const std::vector<TrainingExample>& dataset,
                                               const SynthesisConfig& config);

[[nodiscard]] std::optional<ExactPatternMatch> detect_exact_pattern(const BooleanCascade& initial,
                                                                    const std::vector<TrainingExample>& dataset,
                                                                    std::size_t preferred_gate_count);

[[nodiscard]] std::vector<ExactPatternMatch> exact_pattern_registry(const BooleanCascade& initial,
                                                                    const std::vector<TrainingExample>& dataset,
                                                                    std::size_t preferred_gate_count);

[[nodiscard]] SynthesisResult synthesize_local_jit(BooleanCascade initial,
                                                   const std::vector<TrainingExample>& dataset,
                                                   const SynthesisConfig& config,
                                                   JitCompileOptions jit_options = {});

[[nodiscard]] SynthesisResult synthesize_multigate(std::size_t input_count,
                                                   std::size_t gate_count,
                                                   const std::vector<TrainingExample>& dataset,
                                                   const SynthesisConfig& config);

[[nodiscard]] ResidualSynthesisResult synthesize_with_residual(BooleanCascade base_initial,
                                                               const std::vector<TrainingExample>& dataset,
                                                               const SynthesisConfig& config,
                                                               const ResidualPlannerConfig& planner = {});

[[nodiscard]] std::vector<TieredSynthesisCandidate> synthesize_tiered(BooleanCascade initial,
                                                                      const std::vector<TrainingExample>& dataset,
                                                                      const SynthesisConfig& config,
                                                                      const SolverPolicy& policy = {},
                                                                      std::vector<TierDiagnostics>* diagnostics = nullptr);

[[nodiscard]] SolverPolicy default_solver_policy_for_task(TaskKind kind);

void mutate_topology(BooleanCascade& cascade, std::mt19937_64& rng);

} // namespace sle
