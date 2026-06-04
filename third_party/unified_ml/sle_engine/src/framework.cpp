#include "sle/framework.hpp"

#include <utility>

namespace sle {

SolverPolicy default_solver_policy_for_task(TaskKind kind) {
    SolverPolicy policy;
    switch (kind) {
        case TaskKind::BooleanFunction:
            policy.tiers = {
                {SynthesisTier::Exact, true, 0, true},
                {SynthesisTier::Local, true, 0, true},
                {SynthesisTier::MonteCarloTreeSearch, true, 0, false},
            };
            policy.residual_policy = ResidualPolicyMode::FallbackOnly;
            break;
        case TaskKind::ConstrainedBooleanFunction:
            policy.tiers = {
                {SynthesisTier::Exact, true, 0, true},
                {SynthesisTier::MonteCarloTreeSearch, true, 0, false},
                {SynthesisTier::Local, true, 0, false},
            };
            policy.residual_policy = ResidualPolicyMode::AlwaysRefine;
            break;
        case TaskKind::StochasticBitstream:
            policy.tiers = {
                {SynthesisTier::Local, true, 0, false},
                {SynthesisTier::MonteCarloTreeSearch, true, 0, false},
            };
            policy.residual_policy = ResidualPolicyMode::AlwaysRefine;
            break;
    }
    return policy;
}

std::vector<TrainingExample> Dataset::as_training_examples() const {
    std::vector<TrainingExample> out;
    out.reserve(samples_.size());
    for (const auto& sample : samples_) out.push_back(TrainingExample{sample.inputs, sample.target});
    return out;
}

BitVector FrameworkModel::predict(const std::vector<BitVector>& inputs) const {
    return run_full_engine(model_, inputs);
}

FrameworkModel Trainer::fit(const Dataset& dataset, const HardLogicContract& contract) {
    auto examples = dataset.as_training_examples();
    if (config_.engine.solver_policy.tiers.empty()) {
        config_.engine.solver_policy = default_solver_policy_for_task(dataset.kind());
    }
    auto trained = train_full_engine(examples, config_.engine, contract);
    return FrameworkModel(std::move(trained.model));
}

Metrics Trainer::evaluate(const FrameworkModel& model,
                          const Dataset& dataset,
                          const HardLogicContract& contract) const {
    Metrics metrics;
    const auto examples = dataset.as_training_examples();
    const auto fitness = model.raw().residual.has_value()
        ? evaluate_fitness_residual(model.raw().base, *model.raw().residual, examples, config_.engine.synthesis)
        : evaluate_fitness(model.raw().base, examples, config_.engine.synthesis);
    metrics.output_fitness = fitness.output_fitness;
    metrics.total_fitness = fitness.total;

    for (const auto& sample : dataset.samples()) {
        auto out = run_full_engine(model.raw(), sample.inputs);
        if (contract.forbidden.size() == out.size()) out = apply_forbidden_filter(out, contract.forbidden);
        if (out.hamming_distance(sample.target) != 0) ++metrics.errors;
    }

    return metrics;
}

Dataset make_boolean_dataset(const std::vector<TrainingExample>& examples, TaskKind kind) {
    std::vector<Sample> samples;
    samples.reserve(examples.size());
    for (const auto& ex : examples) samples.push_back(Sample{ex.inputs, ex.target});
    return Dataset(std::move(samples), kind);
}

TrainerConfig default_trainer_config_for_task(TaskKind kind) {
    TrainerConfig cfg;
    cfg.engine.solver_policy = default_solver_policy_for_task(kind);
    return cfg;
}

} // namespace sle
