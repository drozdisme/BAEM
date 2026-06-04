#include "sle/engine.hpp"

#include "sle/fast_ops.hpp"
#include "sle/validation.hpp"

#include <stdexcept>

namespace sle {

BitVector BooleanCascade::evaluate(const std::vector<BitVector>& inputs) const {
    if (inputs.size() != input_count_) throw std::invalid_argument("BooleanCascade input count mismatch");
    if (inputs.empty()) throw std::invalid_argument("BooleanCascade requires at least one input");
    const auto validation = validate_cascade(*this);
    if (!validation.ok) {
        throw std::invalid_argument("BooleanCascade invalid topology at gate " + std::to_string(validation.gate_index) + ": " + validation.message);
    }

    auto nodes = inputs;
    for (const auto& gate : gates_) {
        if (gate.a >= nodes.size() || gate.b >= nodes.size() || gate.c >= nodes.size()) {
            throw std::out_of_range("BooleanCascade gate index out of range");
        }
        nodes.push_back(ternary_apply_fast(nodes[gate.a], nodes[gate.b], nodes[gate.c], gate.mask));
    }
    return nodes.back();
}

BitVector Engine::run(const std::vector<BitVector>& inputs, const BitVector* residual) const {
    std::vector<BitVector> mixed_inputs;
    mixed_inputs.reserve(inputs.size());
    for (const auto& input : inputs) mixed_inputs.push_back(mixer_.mix(input));

    BitVector output = cascade_.evaluate(mixed_inputs);
    if (residual != nullptr) output = residual_apply(output, *residual);
    output = apply_forbidden_filter(output, contract_.forbidden);
    return output;
}

} // namespace sle
