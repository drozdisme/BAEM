#include "sle/synthesis.hpp"
#include "sle/perception.hpp"
#include "sle/topology_search.hpp"
#include "sle/validation.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sle {

namespace {

std::string tier_name(SynthesisTier tier) {
    switch (tier) {
        case SynthesisTier::Exact: return "exact";
        case SynthesisTier::Local: return "local";
        case SynthesisTier::MonteCarloTreeSearch: return "mcts";
        case SynthesisTier::Residual: return "residual";
    }
    return "unknown";
}

BitVector evaluate_without_contract(const BooleanCascade& cascade, const std::vector<BitVector>& inputs) {
    return cascade.evaluate(inputs);
}

BitVector evaluate_residual_pair(const BooleanCascade& base,
                                 const BooleanCascade& residual,
                                 const std::vector<BitVector>& inputs) {
    const auto base_out = base.evaluate(inputs);
    const auto residual_out = residual.evaluate(inputs);
    return residual_apply(base_out, residual_out);
}

std::size_t total_bits(const std::vector<TrainingExample>& dataset) {
    std::size_t total = 0;
    for (const auto& ex : dataset) total += ex.target.size();
    return total;
}

FitnessBreakdown build_fitness(std::size_t error_bits,
                               std::size_t bit_total,
                               double mdl_model_bits,
                               double computational_cost,
                               double critical_path_cost,
                               const SynthesisConfig& config) {
    const double denom = static_cast<double>(std::max<std::size_t>(1, bit_total));
    const double output_fitness = 1.0 - static_cast<double>(error_bits) / denom;
    const double mdl_residual_bits = static_cast<double>(error_bits) * config.mdl_error_bit_cost;
    const double mdl_total_bits = mdl_model_bits + mdl_residual_bits;
    const double mdl_term = config.use_mdl
        ? mdl_total_bits / denom
        : config.complexity_weight * (mdl_model_bits / std::max(1.0, config.mdl_gate_bits));
    const double compute_term = config.computational_cost_weight * (computational_cost / denom)
        + config.critical_path_weight * critical_path_cost;
    const double complexity_penalty = mdl_term + compute_term;
    return FitnessBreakdown{
        output_fitness,
        complexity_penalty,
        mdl_model_bits,
        mdl_residual_bits,
        computational_cost,
        critical_path_cost,
        output_fitness - complexity_penalty,
    };
}

bool dataset_is_uniform(const std::vector<TrainingExample>& dataset, std::size_t& word_count_out) {
    if (dataset.empty()) return false;
    const auto input_count = dataset.front().inputs.size();
    const auto bit_count = dataset.front().target.size();
    const auto word_count = dataset.front().target.word_count();
    for (const auto& ex : dataset) {
        if (ex.inputs.size() != input_count || ex.target.size() != bit_count || ex.target.word_count() != word_count) return false;
        for (const auto& input : ex.inputs) {
            if (input.size() != bit_count || input.word_count() != word_count) return false;
        }
    }
    word_count_out = word_count;
    return true;
}

} // namespace

double estimate_description_length_bits(const BooleanCascade& cascade,
                                        const SynthesisConfig& config) {
    double bits = 0.0;
    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        const auto& gate = cascade.gates()[gate_index];
        const auto address_bits = config.mdl_topology_bits;
        bits += config.mdl_gate_bits + address_bits * 3.0;
        if (gate.a == gate_index + cascade.input_count()) bits -= 1.0;
        if (gate.b == gate_index + cascade.input_count()) bits -= 1.0;
        if (gate.c == gate_index + cascade.input_count()) bits -= 1.0;
        if (gate.mask == 0x96 || gate.mask == 0xCA || gate.mask == 0xE4) bits -= 4.0;
    }
    return std::max(0.0, bits);
}

double estimate_computational_cost(const BooleanCascade& cascade,
                                   const SynthesisConfig&,
                                   double* critical_path_cost) {
    const std::size_t node_count = cascade.input_count() + cascade.gate_count();
    std::vector<double> depth(node_count, 0.0);
    double total_cost = 0.0;
    double max_depth = 0.0;

    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        const auto& gate = cascade.gates()[gate_index];
        double gate_cost = 1.0;
        if (gate.mask == 0x96) gate_cost = 2.8;
        else if (gate.mask == 0xCA || gate.mask == 0xE4) gate_cost = 2.2;
        else if (gate.mask == 0x00 || gate.mask == 0xFF) gate_cost = 0.2;
        else if (gate.mask == 0xF0 || gate.mask == 0xCC || gate.mask == 0xAA) gate_cost = 0.8;

        const std::size_t output_index = cascade.input_count() + gate_index;
        const double input_depth = std::max({depth[gate.a], depth[gate.b], depth[gate.c]});
        depth[output_index] = input_depth + gate_cost;
        total_cost += gate_cost;
        max_depth = std::max(max_depth, depth[output_index]);
    }

    if (critical_path_cost != nullptr) *critical_path_cost = max_depth;
    return total_cost;
}

FitnessBreakdown evaluate_fitness(const BooleanCascade& cascade,
                                  const std::vector<TrainingExample>& dataset,
                                  const SynthesisConfig& config) {
    if (dataset.empty()) throw std::invalid_argument("dataset must not be empty");

    const auto validation = validate_cascade(cascade);
    if (!validation.ok) {
        return FitnessBreakdown{0.0, 1e9, 1e9, 0.0, 1e9, 1e9, -1e9};
    }

    std::size_t error_bits = 0;
    for (const auto& ex : dataset) {
        auto out = evaluate_without_contract(cascade, ex.inputs);
        error_bits += out.hamming_distance(ex.target);
    }

    const auto bit_total = total_bits(dataset);
    double critical_path_cost = 0.0;
    const double computational_cost = estimate_computational_cost(cascade, config, &critical_path_cost);
    return build_fitness(error_bits,
                         bit_total,
                         estimate_description_length_bits(cascade, config),
                         computational_cost,
                         critical_path_cost,
                         config);
}

FitnessBreakdown evaluate_fitness(const CompiledBooleanCascade& compiled,
                                  const BooleanCascade& cascade,
                                  const std::vector<TrainingExample>& dataset,
                                  const SynthesisConfig& config) {
    if (dataset.empty()) throw std::invalid_argument("dataset must not be empty");

    const auto validation = validate_cascade(cascade);
    if (!validation.ok) {
        return FitnessBreakdown{0.0, 1e9, 1e9, 0.0, 1e9, 1e9, -1e9};
    }

    std::size_t error_bits = 0;
    for (const auto& ex : dataset) {
        auto out = compiled.evaluate(ex.inputs);
        error_bits += out.hamming_distance(ex.target);
    }

    const auto bit_total = total_bits(dataset);
    double critical_path_cost = 0.0;
    const double computational_cost = estimate_computational_cost(cascade, config, &critical_path_cost);
    return build_fitness(error_bits,
                         bit_total,
                         estimate_description_length_bits(cascade, config),
                         computational_cost,
                         critical_path_cost,
                         config);
}

FitnessBreakdown evaluate_fitness_residual(const BooleanCascade& base,
                                           const BooleanCascade& residual,
                                           const std::vector<TrainingExample>& dataset,
                                           const SynthesisConfig& config) {
    if (dataset.empty()) throw std::invalid_argument("dataset must not be empty");

    const auto base_validation = validate_cascade(base);
    const auto residual_validation = validate_cascade(residual);
    if (!base_validation.ok || !residual_validation.ok) {
        return FitnessBreakdown{0.0, 1e9, 1e9, 0.0, 1e9, 1e9, -1e9};
    }

    std::size_t error_bits = 0;
    for (const auto& ex : dataset) {
        auto out = evaluate_residual_pair(base, residual, ex.inputs);
        error_bits += out.hamming_distance(ex.target);
    }

    const auto bit_total = total_bits(dataset);
    double base_critical = 0.0;
    double residual_critical = 0.0;
    const auto mdl_bits = estimate_description_length_bits(base, config) + estimate_description_length_bits(residual, config);
    const auto computational_cost = estimate_computational_cost(base, config, &base_critical)
        + estimate_computational_cost(residual, config, &residual_critical);
    return build_fitness(error_bits,
                         bit_total,
                         mdl_bits,
                         computational_cost,
                         std::max(base_critical, residual_critical),
                         config);
}

SynthesisResult synthesize_local(BooleanCascade initial,
                                 const std::vector<TrainingExample>& dataset,
                                 const SynthesisConfig& config) {
    if (initial.gate_count() == 0) throw std::invalid_argument("initial cascade must contain at least one gate");

    auto exact = synthesize_exact(initial, dataset, config);
    if (exact.fitness.output_fitness >= config.target_output_fitness) {
        return exact;
    }

    const auto perception = summarize_bitstreams(dataset);
    return synthesize_local_greedy(std::move(initial), dataset, config, nullptr, &perception, nullptr);
}

SynthesisResult synthesize_exact(BooleanCascade initial,
                                 const std::vector<TrainingExample>& dataset,
                                 const SynthesisConfig& config) {
    if (initial.gate_count() == 0) throw std::invalid_argument("initial cascade must contain at least one gate");

    if (initial.gate_count() != 1 || dataset.empty()) {
        return SynthesisResult{std::move(initial), evaluate_fitness(initial, dataset, config), 0};
    }

    std::uint8_t inferred_mask = 0;
    for (const auto& ex : dataset) {
        if (ex.inputs.size() < 3 || ex.target.size() != 1) {
            return SynthesisResult{std::move(initial), evaluate_fitness(initial, dataset, config), 0};
        }

        const std::uint8_t idx = static_cast<std::uint8_t>((ex.inputs[0].get(0) ? 0x4U : 0U)
            | (ex.inputs[1].get(0) ? 0x2U : 0U)
            | (ex.inputs[2].get(0) ? 0x1U : 0U));
        if (ex.target.get(0)) inferred_mask = static_cast<std::uint8_t>(inferred_mask | static_cast<std::uint8_t>(1U << idx));
    }

    auto candidate = initial;
    candidate.mutable_gates().front().mask = inferred_mask;
    auto fitness = evaluate_fitness(candidate, dataset, config);
    return SynthesisResult{std::move(candidate), fitness, fitness.output_fitness >= config.target_output_fitness ? 1U : 0U};
}

std::optional<ExactPatternMatch> detect_exact_pattern(const BooleanCascade& initial,
                                                      const std::vector<TrainingExample>& dataset,
                                                      std::size_t preferred_gate_count) {
    auto patterns = exact_pattern_registry(initial, dataset, preferred_gate_count);
    if (patterns.empty()) return std::nullopt;
    return patterns.front();
}

std::vector<ExactPatternMatch> exact_pattern_registry(const BooleanCascade& initial,
                                                      const std::vector<TrainingExample>& dataset,
                                                      std::size_t preferred_gate_count) {
    std::vector<ExactPatternMatch> patterns;
    if (dataset.empty()) return patterns;

    if (initial.gate_count() == 1 && dataset.front().target.size() == 1 && dataset.front().inputs.size() >= 3) {
        auto exact = synthesize_exact(initial, dataset, SynthesisConfig{});
        if (exact.fitness.output_fitness >= 1.0) {
            patterns.push_back(ExactPatternMatch{ExactPatternKind::TruthTableMask, exact.cascade, "truth-table-mask", "single-gate truth-table exact match"});
        }
    }

    if (!dataset.empty() && dataset.front().target.size() == 1) {
        bool identity = true;
        for (const auto& ex : dataset) {
            if (ex.inputs.empty() || ex.inputs.front().size() != ex.target.size() || ex.inputs.front().to_string() != ex.target.to_string()) {
                identity = false;
                break;
            }
        }
        if (identity && !dataset.front().inputs.empty()) {
            BooleanCascade cascade(dataset.front().inputs.size());
            cascade.add_gate({0, 0, 0, 0xF0});
            patterns.push_back(ExactPatternMatch{ExactPatternKind::IdentityPassThrough, std::move(cascade), "identity-pass-through", "input[0] passes directly to output"});
        }
    }

    if (!dataset.empty() && dataset.front().inputs.size() == 3 && dataset.front().target.size() == 1) {
        bool mux = true;
        for (const auto& ex : dataset) {
            const bool expected = ex.inputs[0].get(0) ? ex.inputs[2].get(0) : ex.inputs[1].get(0);
            if (expected != ex.target.get(0)) {
                mux = false;
                break;
            }
        }
        if (mux) {
            BooleanCascade cascade(3);
            cascade.add_gate({0, 1, 2, 0xCA});
            patterns.push_back(ExactPatternMatch{ExactPatternKind::SimpleMux, std::move(cascade), "simple-mux", "recognized 3-input mux"});
        }
    }

    const auto input_count = dataset.front().inputs.size();
    if (input_count == 3 && preferred_gate_count >= 2) {
        bool xor3 = true;
        for (const auto& ex : dataset) {
            if (ex.inputs.size() != 3 || ex.target.size() != 1) {
                xor3 = false;
                break;
            }
            const bool expected = ex.inputs[0].get(0) ^ ex.inputs[1].get(0) ^ ex.inputs[2].get(0);
            if (expected != ex.target.get(0)) {
                xor3 = false;
                break;
            }
        }
        if (xor3) {
            BooleanCascade cascade(input_count);
            cascade.add_gate({0, 1, 1, 0x66});
            cascade.add_gate({input_count, 2, 2, 0x66});
            while (cascade.gate_count() < preferred_gate_count) {
                const std::size_t prev = input_count + cascade.gate_count() - 1;
                cascade.add_gate({prev, prev, prev, 0xF0});
            }
            patterns.push_back(ExactPatternMatch{ExactPatternKind::XorChain3, std::move(cascade), "xor-chain-3", "recognized 3-input xor chain"});
        }
    }

    if (!dataset.empty() && dataset.front().target.size() == 1 && dataset.front().inputs.size() >= 2) {
        const auto input_count_n = dataset.front().inputs.size();
        bool parity = true;
        for (const auto& ex : dataset) {
            bool expected = false;
            for (const auto& in : ex.inputs) expected ^= in.get(0);
            if (expected != ex.target.get(0)) {
                parity = false;
                break;
            }
        }
        if (parity) {
            BooleanCascade cascade(input_count_n);
            cascade.add_gate({0, 1, 1, 0x66});
            for (std::size_t i = 2; i < input_count_n; ++i) {
                const std::size_t prev = input_count_n + cascade.gate_count() - 1;
                cascade.add_gate({prev, i, i, 0x66});
            }
            patterns.push_back(ExactPatternMatch{input_count_n == 3 ? ExactPatternKind::XorChainN : ExactPatternKind::ParityFamily,
                                                 std::move(cascade),
                                                 "parity-family",
                                                 "recognized xor/parity family"});
        }
    }

    return patterns;
}

SynthesisResult synthesize_local_jit(BooleanCascade initial,
                                     const std::vector<TrainingExample>& dataset,
                                     const SynthesisConfig& config,
                                     JitCompileOptions jit_options) {
    if (initial.gate_count() == 0) throw std::invalid_argument("initial cascade must contain at least one gate");
    if (dataset.empty()) throw std::invalid_argument("dataset must not be empty");

    std::size_t uniform_word_count = 0;
    if (!dataset_is_uniform(dataset, uniform_word_count)) {
        throw std::invalid_argument("synthesize_local_jit requires uniform dataset bit width");
    }
    if (jit_options.word_count == 0) jit_options.word_count = uniform_word_count;
    if (jit_options.word_count != uniform_word_count) {
        throw std::invalid_argument("jit word_count must match dataset word count");
    }

    if (jit_options.preferred_backend == JitBackend::Avx512Ternlog) {
        try {
            auto probe = compile_boolean_cascade(initial, jit_options);
            if (probe.patch_points().size() != initial.gate_count()) jit_options.preferred_backend = JitBackend::ScalarGpr;
        } catch (...) {
            jit_options.preferred_backend = JitBackend::ScalarGpr;
        }
    }

    std::mt19937_64 rng(config.seed);
    auto best_cascade = std::move(initial);

    // Working state for simulated annealing.
    // CompiledBooleanCascade is move-only, so we keep a single compiled
    // instance (cur_compiled) and mutate it in-place via patch_gate_mask.
    // best_cascade / best_fitness are cheap to copy and track the global best.
    auto cur_cascade = best_cascade;
    auto cur_compiled = compile_boolean_cascade(cur_cascade, jit_options);
    auto cur_fitness  = evaluate_fitness(cur_compiled, cur_cascade, dataset, config);
    auto best_fitness = cur_fitness;
    std::size_t accepted = 0;

    std::uniform_int_distribution<std::size_t> gate_dist(0, cur_cascade.gate_count() - 1);
    std::uniform_int_distribution<int> bit_dist(0, 7);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    // Simulated annealing schedule: high temperature early (allows uphill moves),
    // cools to near-zero so final phase is pure greedy.
    // T_start chosen so ~30% of 0.01-delta moves are accepted at iter=0.
    const double T_start = 0.03;
    const double T_end   = 1e-5;
    const double inv_iters = 1.0 / static_cast<double>(std::max<std::size_t>(1, config.iterations));

    for (std::size_t iter = 0; iter < config.iterations; ++iter) {
        const double epoch = static_cast<double>(iter) * inv_iters;
        const double T = T_start * std::pow(T_end / T_start, epoch);

        const auto gate_index = gate_dist(rng);
        const auto old_mask = cur_cascade.mutable_gates()[gate_index].mask;
        const auto new_mask = static_cast<std::uint8_t>(old_mask ^ static_cast<std::uint8_t>(1U << bit_dist(rng)));

        cur_cascade.mutable_gates()[gate_index].mask = new_mask;
        cur_compiled.patch_gate_mask(gate_index, new_mask);

        const auto candidate_fitness = evaluate_fitness(cur_compiled, cur_cascade, dataset, config);
        const double delta = candidate_fitness.total - cur_fitness.total;

        bool accept = delta >= 0.0;
        if (!accept && T > 1e-9) {
            accept = prob_dist(rng) < std::exp(delta / T);
        }

        if (accept) {
            cur_fitness = candidate_fitness;
            ++accepted;
            // Track global best separately
            if (cur_fitness.total > best_fitness.total) {
                best_fitness = cur_fitness;
                best_cascade = cur_cascade;
                // best_compiled not stored: CompiledBooleanCascade is
                // move-only; best_cascade suffices to recompile if needed.
            }
        } else {
            cur_cascade.mutable_gates()[gate_index].mask = old_mask;
            cur_compiled.patch_gate_mask(gate_index, old_mask);
        }

        // Early exit if perfect solution found
        if (best_fitness.output_fitness >= config.target_output_fitness) break;
    }

    return SynthesisResult{best_cascade, best_fitness, accepted};
}

SynthesisResult synthesize_multigate(std::size_t input_count,
                                     std::size_t gate_count,
                                     const std::vector<TrainingExample>& dataset,
                                     const SynthesisConfig& config) {
    if (gate_count == 0) throw std::invalid_argument("gate_count must be > 0");
    BooleanCascade cascade(input_count);
    for (std::size_t i = 0; i < gate_count; ++i) {
        const std::size_t available = input_count + i;
        const std::size_t a = available >= 1 ? 0 : 0;
        const std::size_t b = available >= 2 ? std::min<std::size_t>(1, available - 1) : 0;
        const std::size_t c = available >= 3 ? std::min<std::size_t>(2, available - 1) : 0;
        cascade.add_gate({a, b, c, 0x00});
    }

    auto result = synthesize_local(std::move(cascade), dataset, config);
    while (result.cascade.gate_count() < gate_count) {
        result.cascade.add_gate({0, 0, 0, 0x00});
    }
    return result;
}

ResidualSynthesisResult synthesize_with_residual(BooleanCascade base_initial,
                                                 const std::vector<TrainingExample>& dataset,
                                                 const SynthesisConfig& config,
                                                 const ResidualPlannerConfig& planner) {
    auto build_residual_dataset = [&](const BooleanCascade& base_cascade) {
        std::vector<TrainingExample> residual_dataset;
        residual_dataset.reserve(dataset.size());
        for (const auto& ex : dataset) {
            const auto base_out = base_cascade.evaluate(ex.inputs);
            residual_dataset.push_back(TrainingExample{ex.inputs, residual_target(ex.target, base_out)});
        }
        return residual_dataset;
    };

    auto evaluate_pair = [&](const BooleanCascade& base_cascade, std::uint64_t seed_bias = 0ULL) {
        auto residual_dataset = build_residual_dataset(base_cascade);
        BooleanCascade residual(base_cascade.input_count());
        residual.add_gate({0, 1, 2, 0x00});

        SynthesisResult residual_result{residual, evaluate_fitness(residual, residual_dataset, config), 0};
        std::vector<ResidualPlanStep> plan_steps;

        for (const auto& step_cfg : planner.tiers) {
            if (!step_cfg.enabled) continue;
            auto residual_config = config;
            residual_config.seed ^= seed_bias;
            if (step_cfg.iterations_override != 0) residual_config.iterations = step_cfg.iterations_override;

            SynthesisResult step_result = residual_result;
            switch (step_cfg.tier) {
                case SynthesisTier::Exact:
                    step_result = synthesize_exact(residual_result.cascade, residual_dataset, residual_config);
                    break;
                case SynthesisTier::Local:
                    step_result = synthesize_local(residual_result.cascade, residual_dataset, residual_config);
                    break;
                case SynthesisTier::MonteCarloTreeSearch: {
                    const auto perception = summarize_bitstreams(residual_dataset);
                    CompilationCache cache({1, JitBackend::ScalarGpr}, 256);
                    step_result = synthesize_mcts(residual_result.cascade, residual_dataset, residual_config, nullptr, &perception, &cache);
                    break;
                }
                case SynthesisTier::Residual:
                    continue;
            }

            const auto final_fit = evaluate_fitness_residual(base_cascade, step_result.cascade, dataset, config);
            plan_steps.push_back(ResidualPlanStep{tier_name(step_cfg.tier), step_result, final_fit, "residual step via " + tier_name(step_cfg.tier)});
            if (final_fit.output_fitness > evaluate_fitness_residual(base_cascade, residual_result.cascade, dataset, config).output_fitness
                || final_fit.total > evaluate_fitness_residual(base_cascade, residual_result.cascade, dataset, config).total) {
                residual_result = step_result;
            }
            if (step_cfg.stop_on_target && final_fit.output_fitness >= config.target_output_fitness) break;
        }

        if (residual_result.cascade.gate_count() > 0) {
            const std::uint8_t masks[] = {residual_result.cascade.gates().front().mask, 0x96, 0xCA, 0xE8, 0xFE, 0x00, 0xFF};
            auto best_residual = residual_result;
            auto best_final = evaluate_fitness_residual(base_cascade, residual_result.cascade, dataset, config);
            for (auto mask : masks) {
                auto candidate = residual_result.cascade;
                candidate.mutable_gates().front().mask = mask;
                auto final_fit = evaluate_fitness_residual(base_cascade, candidate, dataset, config);
                if (final_fit.output_fitness > best_final.output_fitness
                    || (final_fit.output_fitness == best_final.output_fitness && final_fit.total > best_final.total)) {
                    best_residual.cascade = std::move(candidate);
                    best_residual.fitness = evaluate_fitness(best_residual.cascade, residual_dataset, config);
                    best_final = final_fit;
                }
            }
            residual_result = std::move(best_residual);
            return std::tuple<std::vector<TrainingExample>, SynthesisResult, FitnessBreakdown, std::vector<ResidualPlanStep>>{
                std::move(residual_dataset), std::move(residual_result), best_final, std::move(plan_steps)};
        }

        auto final_fitness = evaluate_fitness_residual(base_cascade, residual_result.cascade, dataset, config);
        return std::tuple<std::vector<TrainingExample>, SynthesisResult, FitnessBreakdown, std::vector<ResidualPlanStep>>{
            std::move(residual_dataset), std::move(residual_result), final_fitness, std::move(plan_steps)};
    };

    auto base = synthesize_local(base_initial, dataset, config);
    if (base.cascade.gate_count() == 0) {
        base.cascade.add_gate({0, 0, 0, 0x00});
        base.fitness = evaluate_fitness(base.cascade, dataset, config);
    }

    auto [best_residual_dataset, best_residual, best_final_fitness, best_steps] = evaluate_pair(base.cascade);
    auto best_base = base;

    if (base_initial.gate_count() > 0 && base_initial.gate_count() == 1) {
        const std::uint8_t candidate_masks[] = {0xFE, 0x96, 0xCA, 0xE8, 0x00, 0xFF};
        for (std::size_t i = 0; i < std::size(candidate_masks); ++i) {
            auto candidate_base_cascade = base_initial;
            candidate_base_cascade.mutable_gates().front().mask = candidate_masks[i];
            auto candidate_base_fitness = evaluate_fitness(candidate_base_cascade, dataset, config);
            auto [candidate_residual_dataset, candidate_residual, candidate_final, candidate_steps] =
                evaluate_pair(candidate_base_cascade, 0xA5A5A5A500000000ULL ^ static_cast<std::uint64_t>(i));

            if (candidate_final.output_fitness > best_final_fitness.output_fitness
                || (candidate_final.output_fitness == best_final_fitness.output_fitness
                    && candidate_final.total > best_final_fitness.total)) {
                best_base.cascade = std::move(candidate_base_cascade);
                best_base.fitness = candidate_base_fitness;
                best_residual_dataset = std::move(candidate_residual_dataset);
                best_residual = std::move(candidate_residual);
                best_final_fitness = candidate_final;
                best_steps = std::move(candidate_steps);
            }
        }
    }

    return ResidualSynthesisResult{best_base, best_residual, best_final_fitness,
        "residual refinement evaluated base and residual candidates", std::move(best_steps)};
}

std::vector<TieredSynthesisCandidate> synthesize_tiered(BooleanCascade initial,
                                                        const std::vector<TrainingExample>& dataset,
                                                        const SynthesisConfig& config,
                                                        const SolverPolicy& policy,
                                                        std::vector<TierDiagnostics>* diagnostics) {
    std::vector<TieredSynthesisCandidate> candidates;
    std::vector<TierDiagnostics> local_diagnostics;

    for (const auto& tier_cfg : policy.tiers) {
        if (!tier_cfg.enabled) continue;

        auto tier_config = config;
        if (tier_cfg.iterations_override != 0) tier_config.iterations = tier_cfg.iterations_override;

        TieredSynthesisCandidate candidate;
        candidate.tier = tier_cfg.tier;
        candidate.diagnostics.tier = tier_cfg.tier;
        candidate.diagnostics.tier_name = tier_name(tier_cfg.tier);
        candidate.diagnostics.executed = true;

        switch (tier_cfg.tier) {
            case SynthesisTier::Exact: {
                if (auto pattern = detect_exact_pattern(initial, dataset, initial.gate_count())) {
                    candidate.result.cascade = pattern->cascade;
                    candidate.result.fitness = evaluate_fitness(candidate.result.cascade, dataset, tier_config);
                    candidate.result.accepted_mutations = 1;
                    candidate.diagnostics.summary = pattern->name + ": " + pattern->description;
                } else {
                    candidate.result = synthesize_exact(initial, dataset, tier_config);
                    candidate.diagnostics.summary = "exact truth-table solver";
                }
                break;
            }
            case SynthesisTier::Local: {
                candidate.result = synthesize_local(initial, dataset, tier_config);
                candidate.diagnostics.summary = "local greedy tier";
                break;
            }
            case SynthesisTier::MonteCarloTreeSearch: {
                const auto perception = summarize_bitstreams(dataset);
                candidate.result = synthesize_mcts(initial, dataset, tier_config, nullptr, &perception, nullptr);
                candidate.diagnostics.summary = "mcts planning tier";
                break;
            }
            case SynthesisTier::Residual:
                continue;
        }

        candidate.diagnostics.fitness = candidate.result.fitness;
        candidate.diagnostics.accepted_mutations = candidate.result.accepted_mutations;
        candidate.diagnostics.reached_target = candidate.result.fitness.output_fitness >= tier_config.target_output_fitness;
        local_diagnostics.push_back(candidate.diagnostics);
        candidates.push_back(std::move(candidate));

        if (!candidates.empty() && tier_cfg.stop_on_target && candidates.back().diagnostics.reached_target) break;
    }

    if (diagnostics != nullptr) *diagnostics = std::move(local_diagnostics);

    return candidates;
}

} // namespace sle