#include "sle/full_engine.hpp"

namespace sle {

namespace {

FitnessBreakdown evaluate_full_model(const FullEngineModel& model,
                                     const std::vector<TrainingExample>& dataset,
                                     double complexity_weight) {
    if (model.residual.has_value()) {
        auto config = model.config.synthesis;
        config.complexity_weight = complexity_weight;
        return evaluate_fitness_residual(model.base, *model.residual, dataset, config);
    }
    auto config = model.config.synthesis;
    config.complexity_weight = complexity_weight;
    return evaluate_fitness(model.base, dataset, config);
}

double contract_penalty(const FullEngineModel& model,
                        const std::vector<TrainingExample>& dataset,
                        const HardLogicContract& contract,
                        bool hlc_aware) {
    if (!hlc_aware || dataset.empty() || contract.forbidden.size() != dataset.front().target.size()) return 0.0;

    std::size_t violations = 0;
    for (const auto& ex : dataset) {
        auto raw = model.residual.has_value()
            ? residual_apply(model.base.evaluate(ex.inputs), model.residual->evaluate(ex.inputs))
            : model.base.evaluate(ex.inputs);
        violations += raw.bit_and(contract.forbidden).popcount();
    }

    return static_cast<double>(violations) / static_cast<double>(dataset.size() * dataset.front().target.size() + 1);
}

FullEngineTrainResult finalize_candidate(const FullEngineModel& model,
                                         const std::vector<TrainingExample>& dataset,
                                         const HardLogicContract& contract,
                                         bool hlc_aware,
                                         SynthesisTier tier,
                                         std::vector<TierDiagnostics> tier_diagnostics = {},
                                         std::string summary = {}) {
    auto fitness = evaluate_full_model(model, dataset, model.config.synthesis.complexity_weight);
    fitness.total -= contract_penalty(model, dataset, contract, hlc_aware);
    return FullEngineTrainResult{model, fitness, tier, std::move(tier_diagnostics), std::move(summary)};
}

bool better_candidate(const FullEngineTrainResult& lhs,
                      const FullEngineTrainResult& rhs) {
    if (lhs.fitness.output_fitness != rhs.fitness.output_fitness) {
        return lhs.fitness.output_fitness > rhs.fitness.output_fitness;
    }
    return lhs.fitness.total > rhs.fitness.total;
}

} // namespace

FullEngineTrainResult train_full_engine(const std::vector<TrainingExample>& dataset,
                                        const FullEngineConfig& config,
                                        HardLogicContract contract) {
    if (dataset.empty()) {
        throw std::invalid_argument("dataset must not be empty");
    }

    BooleanCascade initial(dataset.front().inputs.size());
    for (std::size_t i = 0; i < config.gate_count; ++i) {
        const std::size_t available = initial.input_count() + i;
        const std::size_t a = available >= 1 ? 0 : 0;
        const std::size_t b = available >= 2 ? std::min<std::size_t>(1, available - 1) : 0;
        const std::size_t c = available >= 3 ? std::min<std::size_t>(2, available - 1) : 0;
        initial.add_gate({a, b, c, 0x00});
    }

    std::vector<FullEngineTrainResult> candidates;
    std::vector<TierDiagnostics> tier_diagnostics;
    for (auto& candidate : synthesize_tiered(initial, dataset, config.synthesis, config.solver_policy, &tier_diagnostics)) {
        candidates.push_back(finalize_candidate(
            FullEngineModel{candidate.result.cascade, std::nullopt, contract, config},
            dataset,
            contract,
            config.hlc_aware,
            candidate.tier,
            tier_diagnostics,
            candidate.diagnostics.summary));
    }

    if (candidates.empty()) {
        throw std::invalid_argument("train_full_engine: no synthesis tiers enabled");
    }

    auto best = candidates.front();
    for (std::size_t i = 1; i < candidates.size(); ++i) {
        if (better_candidate(candidates[i], best)) best = candidates[i];
    }

    if (config.solver_policy.enable_topology_mutation && best.fitness.output_fitness + 1e-12 < config.synthesis.target_output_fitness) {
        // Multi-restart topology mutation: mutate the wiring, then re-run synthesis
        // from the mutated topology so the mask search has a fresh starting point.
        const std::size_t topology_restarts = std::min<std::size_t>(4, 1 + config.gate_count / 3);
        std::mt19937_64 rng(config.synthesis.seed ^ 0x9E3779B97F4A7C15ULL);
        auto restart_config = config.synthesis;
        restart_config.iterations = std::max<std::size_t>(512, config.synthesis.iterations / 2);

        for (std::size_t r = 0; r < topology_restarts; ++r) {
            if (best.fitness.output_fitness + 1e-12 >= config.synthesis.target_output_fitness) break;

            auto mutated = best.model.base;
            const std::size_t n_muts = 1 + (r % std::max<std::size_t>(1, mutated.gate_count()));
            for (std::size_t m = 0; m < n_muts; ++m) mutate_topology(mutated, rng);

            restart_config.seed = config.synthesis.seed ^ (0xDEADC0DE00ULL + r * 0x1234567ULL);
            std::vector<TierDiagnostics> restart_diag;
            for (auto& rc : synthesize_tiered(mutated, dataset, restart_config, config.solver_policy, &restart_diag)) {
                auto rr = finalize_candidate(
                    FullEngineModel{rc.result.cascade, std::nullopt, contract, config},
                    dataset, contract, config.hlc_aware, rc.tier, restart_diag,
                    "topology-mutation restart " + std::to_string(r));
                if (better_candidate(rr, best)) best = std::move(rr);
            }
        }
    }

    const bool run_residual = config.solver_policy.residual_policy == ResidualPolicyMode::AlwaysRefine
        || (config.solver_policy.residual_policy == ResidualPolicyMode::FallbackOnly
            && best.fitness.output_fitness + 1e-12 < config.synthesis.target_output_fitness);

    if (run_residual) {
        auto residual = synthesize_with_residual(best.model.base, dataset, config.synthesis, config.residual_planner);
        auto residual_result = finalize_candidate(
            FullEngineModel{residual.base.cascade, residual.residual.cascade, contract, config},
            dataset,
            contract,
            config.hlc_aware,
            SynthesisTier::Residual,
            best.tier_diagnostics,
            residual.summary);
        residual_result.tier_diagnostics.push_back(TierDiagnostics{
            SynthesisTier::Residual,
            "residual",
            true,
            residual.final_fitness.output_fitness >= config.synthesis.target_output_fitness,
            residual.final_fitness,
            residual.residual.accepted_mutations,
            residual.summary,
        });
        if (better_candidate(residual_result, best)) best = std::move(residual_result);
    }

    best.selection_summary = "selected tier=" + std::to_string(static_cast<int>(best.selected_tier))
        + ", fitness=" + std::to_string(best.fitness.total);

    return best;
}

BitVector run_full_engine(const FullEngineModel& model,
                          const std::vector<BitVector>& inputs) {
    auto out = model.base.evaluate(inputs);
    if (model.residual.has_value()) {
        out = residual_apply(out, model.residual->evaluate(inputs));
    }
    if (model.contract.forbidden.size() == out.size()) {
        out = apply_forbidden_filter(out, model.contract.forbidden);
    }
    return out;
}

} // namespace sle