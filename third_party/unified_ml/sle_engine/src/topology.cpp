#include "sle/synthesis.hpp"

#include <random>

namespace sle {

void mutate_topology(BooleanCascade& cascade, std::mt19937_64& rng) {
    if (cascade.gate_count() == 0) return;
    std::uniform_int_distribution<std::size_t> gate_dist(0, cascade.gate_count() - 1);
    const std::size_t gate_index = gate_dist(rng);
    auto& gate = cascade.mutable_gates()[gate_index];
    const std::size_t available_inputs = cascade.input_count() + gate_index;
    if (available_inputs == 0) return;
    std::uniform_int_distribution<std::size_t> idx_dist(0, available_inputs - 1);
    switch (rng() % 3ULL) {
        case 0: gate.a = idx_dist(rng); break;
        case 1: gate.b = idx_dist(rng); break;
        default: gate.c = idx_dist(rng); break;
    }
}

} // namespace sle
