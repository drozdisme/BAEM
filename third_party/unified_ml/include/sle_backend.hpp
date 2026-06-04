#pragma once

#include "autograd/tensor.h"
#include "models/mlp/model.hpp"
#include "models/pinn/neural_network.hpp"
#include "sle/sle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace unified_ml::sle_backend {

using BooleanCascade = ::sle::BooleanCascade;
using CompilationCache = ::sle::CompilationCache;
using CachedTopologyProgram = ::sle::CachedTopologyProgram;
using JitCompileOptions = ::sle::JitCompileOptions;

namespace detail {

inline std::uint8_t mask_from_scalar(double value) {
    const auto clamped = std::clamp(value, 0.0, 255.0);
    return static_cast<std::uint8_t>(std::llround(clamped));
}

inline std::vector<std::uint8_t> tensor_to_gate_masks(const autograd::Tensor& tensor, std::size_t gate_count) {
    std::vector<std::uint8_t> masks(gate_count, 0);
    const auto count = std::min(gate_count, tensor.numel());
    for (std::size_t i = 0; i < count; ++i) {
        masks[i] = mask_from_scalar(tensor.value_flat(i));
    }
    return masks;
}

inline void patch_all_masks_from_tensor(::sle::CompiledBooleanCascade& compiled,
                                        const autograd::Tensor& tensor,
                                        bool synchronize = true) {
    const auto masks = tensor_to_gate_masks(tensor, compiled.gate_count());
    for (std::size_t gate_index = 0; gate_index < masks.size(); ++gate_index) {
        compiled.patch_gate_mask(gate_index, masks[gate_index], false);
    }
    if (synchronize) {
        __builtin___clear_cache(reinterpret_cast<char*>(compiled.raw_code()),
                                reinterpret_cast<char*>(compiled.raw_code() + compiled.code_size()));
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("cpuid" : : "a"(0) : "rbx", "rcx", "rdx", "memory");
#endif
    }
}

} // namespace detail

class TensorBoundCompilationCache {
public:
    explicit TensorBoundCompilationCache(JitCompileOptions options = {}, std::size_t max_entries = 256)
        : cache_(options, max_entries) {}

    [[nodiscard]] CachedTopologyProgram get_or_compile(const BooleanCascade& cascade) {
        return cache_.get_or_compile(cascade);
    }

    [[nodiscard]] CachedTopologyProgram get(const std::string& signature) {
        return cache_.get(signature);
    }

    [[nodiscard]] bool contains(const std::string& signature) const {
        return cache_.contains(signature);
    }

    [[nodiscard]] std::size_t size() const noexcept { return cache_.size(); }
    [[nodiscard]] std::size_t hits() const noexcept { return cache_.hits(); }
    [[nodiscard]] std::size_t misses() const noexcept { return cache_.misses(); }

    void patch_all_masks(const std::string& signature, const autograd::Tensor& tensor, bool synchronize = true) {
        auto program = cache_.get(signature);
        if (!program.compiled) throw std::logic_error("SLE compilation cache returned empty program");
        detail::patch_all_masks_from_tensor(*program.compiled, tensor, synchronize);
    }

    void patch_all_masks(CachedTopologyProgram& program, const autograd::Tensor& tensor, bool synchronize = true) {
        if (!program.compiled) throw std::logic_error("SLE cached topology program is empty");
        detail::patch_all_masks_from_tensor(*program.compiled, tensor, synchronize);
    }

private:
    CompilationCache cache_;
};

inline BooleanCascade distill_to_logic(const mlp::Model& model) {
    auto* mutable_model = const_cast<mlp::Model*>(&model);
    const auto params = mutable_model->parameters();
    if (params.empty()) throw std::invalid_argument("distill_to_logic(mlp::Model): model has no parameters");

    BooleanCascade cascade(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto* tensor = params[i];
        const auto scalar = (tensor && tensor->numel() > 0) ? tensor->value_flat(0) : 0.0;
        const auto mask = detail::mask_from_scalar(std::abs(scalar) * 255.0);
        cascade.add_gate({i, i, i, mask});
    }
    return cascade;
}

inline BooleanCascade distill_to_logic(const pinn::NeuralNetwork& model) {
    auto* mutable_model = const_cast<pinn::NeuralNetwork*>(&model);
    const auto params = mutable_model->parameters();
    if (params.empty()) throw std::invalid_argument("distill_to_logic(pinn::NeuralNetwork): model has no parameters");

    BooleanCascade cascade(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto* tensor = params[i];
        const auto scalar = (tensor && tensor->numel() > 0) ? tensor->value_flat(0) : 0.0;
        const auto mask = detail::mask_from_scalar(std::abs(scalar) * 255.0);
        cascade.add_gate({i, i, i, mask});
    }
    return cascade;
}

} // namespace unified_ml::sle_backend
