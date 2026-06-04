#pragma once
#include "models/mlp/layer.hpp"
#include "core/activations.hpp"

namespace mlp {
// Extended ops for BCE loss
autograd::Tensor exp_tensor(const autograd::Tensor& a);
autograd::Tensor log_tensor(const autograd::Tensor& a);
autograd::Tensor sigmoid_fn(const autograd::Tensor& a);
autograd::Tensor tanh_fn(const autograd::Tensor& a);

class ReLU final : public Layer {
public:
    autograd::Tensor forward(const autograd::Tensor& input) override { return autograd::relu(input); }
    std::vector<autograd::Tensor*> parameters() override { return {}; }
};
class Sigmoid final : public Layer {
public:
    autograd::Tensor forward(const autograd::Tensor& input) override { return core::sigmoid_act(input); }
    std::vector<autograd::Tensor*> parameters() override { return {}; }
};
class Tanh final : public Layer {
public:
    autograd::Tensor forward(const autograd::Tensor& input) override { return core::tanh_act(input); }
    std::vector<autograd::Tensor*> parameters() override { return {}; }
};
} // namespace mlp
