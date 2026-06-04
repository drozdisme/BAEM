#pragma once
#include "autograd/autograd.h"
#include "core/activations.hpp"
#include <string>

namespace deep_onet {
enum class Activation { None, ReLU, Tanh, Sigmoid };

inline autograd::Tensor apply_activation(const autograd::Tensor& x, Activation act) {
    switch (act) {
        case Activation::None:    return x;
        case Activation::ReLU:    return autograd::relu(x);
        case Activation::Tanh:    return core::tanh_act(x);
        case Activation::Sigmoid: return core::sigmoid_act(x);
    }
    throw std::invalid_argument("apply_activation: unknown type");
}
Activation activation_from_string(const std::string& name);
} // namespace deep_onet
