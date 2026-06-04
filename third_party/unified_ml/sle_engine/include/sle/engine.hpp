#pragma once

#include "sle/contracts.hpp"
#include "sle/mixer.hpp"
#include "sle/residual.hpp"
#include "sle/ternary.hpp"

#include <cstdint>
#include <vector>
#include <utility>

namespace sle {

struct TernaryGate {
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::uint8_t mask = 0;
};

class BooleanCascade {
public:
    explicit BooleanCascade(std::size_t input_count = 0) : input_count_(input_count) {}

    void add_gate(TernaryGate gate) { gates_.push_back(gate); }
    [[nodiscard]] std::size_t input_count() const noexcept { return input_count_; }
    [[nodiscard]] std::size_t gate_count() const noexcept { return gates_.size(); }
    [[nodiscard]] const std::vector<TernaryGate>& gates() const noexcept { return gates_; }
    [[nodiscard]] std::vector<TernaryGate>& mutable_gates() noexcept { return gates_; }

    [[nodiscard]] BitVector evaluate(const std::vector<BitVector>& inputs) const;

private:
    std::size_t input_count_ = 0;
    std::vector<TernaryGate> gates_;
};

class Engine {
public:
    Engine(BooleanCascade cascade, BitMixer mixer, HardLogicContract contract)
        : cascade_(std::move(cascade)), mixer_(std::move(mixer)), contract_(std::move(contract)) {}

    [[nodiscard]] BitVector run(const std::vector<BitVector>& inputs, const BitVector* residual = nullptr) const;

private:
    BooleanCascade cascade_;
    BitMixer mixer_;
    HardLogicContract contract_;
};

} // namespace sle
