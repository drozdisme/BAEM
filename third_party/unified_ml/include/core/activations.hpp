#pragma once
// activations.hpp — Shared differentiable activations for MLP, DeepONet, PINN.
//
// Each function builds a Node with:
//   backward_fn — fast raw-double path (normal backward)
//   vjp_fn      — Tensor-arithmetic path (create_graph / 2nd-order)
//
// This level of differentiation is required by PINN (d²u/dx² through layers).
// It has zero overhead for MLP/DeepONet which do not use create_graph.

#include "autograd/tensor.h"
#include <string>

namespace core {

/// Element-wise exp: out[i] = exp(x[i]).
/// Supports first and second-order gradients.
autograd::Tensor exp_act(const autograd::Tensor& x);

/// Element-wise tanh: out[i] = tanh(x[i]).
/// Supports first and second-order gradients (VJP via recursive tanh_act).
autograd::Tensor tanh_act(const autograd::Tensor& x);

/// Element-wise sigmoid: out[i] = 1 / (1 + exp(-x[i])).
/// Supports first and second-order gradients.
autograd::Tensor sigmoid_act(const autograd::Tensor& x);

/// Element-wise GELU approximation via tanh formulation.
/// Good default for transformer FFN blocks.
autograd::Tensor gelu_act(const autograd::Tensor& x);

/// Thin wrapper for autograd::relu (no VJP needed — 2nd derivative is 0).
inline autograd::Tensor relu_act(const autograd::Tensor& x) {
    return autograd::relu(x);
}

} // namespace core
