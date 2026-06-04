// layers.h — Fully connected linear layer: y = x @ W + b
//
// Weight matrix W has shape [in_features, out_features].
// Bias vector  b has shape [1, out_features].
//
// Both W and b have requires_grad = true and participate in the computation
// graph. Xavier uniform initialization is used by default (suitable for tanh).
//
// The forward pass uses autograd::matmul and autograd::add, so gradients flow
// correctly through both first and second-order backward passes.

#pragma once

#include "autograd/autograd.h"
#include "models/pinn/tensor_wrapper.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

namespace pinn {

class Linear {
public:
    /// Construct a Linear layer with Xavier-uniform weight init.
    /// @param in_features   Number of input features (columns of input x).
    /// @param out_features  Number of output features.
    /// @param rng           Random number generator for weight initialization.
    Linear(std::size_t in_features,
           std::size_t out_features,
           std::mt19937& rng)
        : in_features_(in_features)
        , out_features_(out_features)
        , weight_(xavier_uniform(in_features, out_features, rng))
        , bias_(zero_bias(out_features))
    {}

    /// Forward pass: y = x @ W + b
    /// @param input  Shape [batch, in_features] or [1, in_features] for single point.
    /// @return       Shape [batch, out_features].
    autograd::Tensor forward(const autograd::Tensor& input) const
    {
        // x @ W: [batch, in] @ [in, out] → [batch, out]
        autograd::Tensor xw = autograd::matmul(input, weight_);
        // + b: [batch, out] + [1, out] → [batch, out]  (broadcast over batch)
        return xw + bias_;
    }

    /// Return raw pointers to learnable parameters (W and b).
    /// The optimizer uses these to read .grad() and update .data().
    std::vector<autograd::Tensor*> parameters()
    {
        return {&weight_, &bias_};
    }

    /// Zero accumulated gradients on both weight and bias.
    void zero_grad()
    {
        weight_.zero_grad();
        bias_.zero_grad();
    }

    //   Accessors                               

    std::size_t in_features()  const noexcept { return in_features_; }
    std::size_t out_features() const noexcept { return out_features_; }

    const autograd::Tensor& weight() const noexcept { return weight_; }
    const autograd::Tensor& bias()   const noexcept { return bias_; }
    autograd::Tensor&       weight()       noexcept { return weight_; }
    autograd::Tensor&       bias()         noexcept { return bias_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    autograd::Tensor weight_;   // [in_features, out_features], requires_grad=true
    autograd::Tensor bias_;     // [1, out_features],           requires_grad=true
};

}  // namespace pinn
