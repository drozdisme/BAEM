#include "sle/synthesis.hpp"

#include <iostream>

static sle::BitVector bit(bool value) {
    sle::BitVector v(1);
    v.set(0, value);
    return v;
}

int main() {
    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)},  bit(true)},
        {{bit(false), bit(true),  bit(false)}, bit(true)},
        {{bit(false), bit(true),  bit(true)},  bit(true)},
        {{bit(true),  bit(false), bit(false)}, bit(true)},
        {{bit(true),  bit(false), bit(true)},  bit(true)},
        {{bit(true),  bit(true),  bit(false)}, bit(true)},
        {{bit(true),  bit(true),  bit(true)},  bit(false)},
    };

    sle::BooleanCascade base(3);
    base.add_gate({0, 1, 2, 0xFE});

    sle::SynthesisConfig cfg;
    cfg.iterations = 4096;
    auto result = sle::synthesize_with_residual(base, dataset, cfg);

    std::cout << "base mask: 0x" << std::hex << static_cast<int>(result.base.cascade.gates().front().mask) << '\n';
    std::cout << "residual mask: 0x" << static_cast<int>(result.residual.cascade.gates().front().mask) << std::dec << '\n';
    std::cout << "final output fitness: " << result.final_fitness.output_fitness << '\n';
    std::cout << "final total fitness: " << result.final_fitness.total << '\n';
    return 0;
}
