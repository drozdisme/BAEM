#include "sle/synthesis.hpp"

#include <iostream>

static sle::BitVector from_bool(bool value) {
    sle::BitVector v(1);
    v.set(0, value);
    return v;
}

int main() {
    std::vector<sle::TrainingExample> dataset{
        {{from_bool(false), from_bool(false), from_bool(false)}, from_bool(false)},
        {{from_bool(false), from_bool(false), from_bool(true)},  from_bool(true)},
        {{from_bool(false), from_bool(true),  from_bool(false)}, from_bool(true)},
        {{from_bool(false), from_bool(true),  from_bool(true)},  from_bool(false)},
        {{from_bool(true),  from_bool(false), from_bool(false)}, from_bool(true)},
        {{from_bool(true),  from_bool(false), from_bool(true)},  from_bool(false)},
        {{from_bool(true),  from_bool(true),  from_bool(false)}, from_bool(false)},
        {{from_bool(true),  from_bool(true),  from_bool(true)},  from_bool(true)},
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x00});

    sle::SynthesisConfig cfg;
    cfg.iterations = 1024;
    auto result = sle::synthesize_local(cascade, dataset, cfg);

    std::cout << "best mask: 0x" << std::hex << static_cast<int>(result.cascade.gates().front().mask) << std::dec << '\n';
    std::cout << "fitness: " << result.fitness.total << '\n';
    std::cout << "accepted mutations: " << result.accepted_mutations << '\n';
    return 0;
}
