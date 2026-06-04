#include "sle/framework.hpp"

#include <iostream>

static sle::BitVector bit(bool value) {
    sle::BitVector v(1);
    v.set(0, value);
    return v;
}

int main() {
    std::vector<sle::TrainingExample> examples{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    auto dataset = sle::make_boolean_dataset(examples);

    sle::TrainerConfig cfg;
    cfg.engine.gate_count = 4;
    cfg.engine.synthesis.iterations = 4096;
    cfg.engine.solver_policy.residual_policy = sle::ResidualPolicyMode::FallbackOnly;

    sle::Trainer trainer(cfg);
    sle::HardLogicContract contract;
    contract.required = sle::BitVector(1, false);
    contract.forbidden = sle::BitVector(1, false);

    auto model = trainer.fit(dataset, contract);
    auto metrics = trainer.evaluate(model, dataset, contract);

    std::cout << "output_fitness=" << metrics.output_fitness << '\n';
    std::cout << "total_fitness=" << metrics.total_fitness << '\n';
    std::cout << "errors=" << metrics.errors << '\n';
    return 0;
}
