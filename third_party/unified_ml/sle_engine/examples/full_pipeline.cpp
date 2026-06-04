#include "sle/full_engine.hpp"

#include <iostream>

static sle::BitVector bit(bool value) {
    sle::BitVector v(1);
    v.set(0, value);
    return v;
}

int main() {
    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::FullEngineConfig cfg;
    cfg.gate_count = 3;
    cfg.synthesis.iterations = 4096;
    cfg.solver_policy.residual_policy = sle::ResidualPolicyMode::FallbackOnly;
    cfg.solver_policy.enable_topology_mutation = true;

    sle::HardLogicContract hlc;
    hlc.required = sle::BitVector(1, false);
    hlc.forbidden = sle::BitVector(1, false);

    auto trained = sle::train_full_engine(dataset, cfg, hlc);

    std::cout << "final fitness: " << trained.fitness.total << '\n';
    std::cout << "base gates: " << trained.model.base.gate_count() << '\n';
    std::cout << "has residual: " << (trained.model.residual.has_value() ? "yes" : "no") << '\n';
    for (const auto& ex : dataset) {
        auto out = sle::run_full_engine(trained.model, ex.inputs);
        std::cout << out.to_string() << '\n';
    }
    return 0;
}
