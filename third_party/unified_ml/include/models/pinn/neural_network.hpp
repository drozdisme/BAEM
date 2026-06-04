// neural_network.h — Fully connected feedforward neural network.
//
// Architecture: alternating Linear layers and activation functions.
//   Input → [Linear → Act] × (depth-1) → Linear → Output
//
// The network holds Linear layers (with weights/biases that require_grad=true).
//
// In addition to the standard forward(), this class provides analytical
// forward-mode differentiation methods for computing spatial derivatives of
// the network output w.r.t. the input coordinates:
//
//   forward_derivs_1d()  — computes (u, du/dx, d²u/dx²) for 1-D spatial input.
//   forward_derivs_2d()  — computes (u, du/dx, d²u/dx², du/dt) for [x,t] input.
//
// These use forward-mode (Jacobian-vector-product) differentiation, propagating
// derivative information through each layer analytically. All intermediate tensors
// use standard Tensor operations (matmul, mul, add, scalar mult) so gradients
// w.r.t. network PARAMETERS flow correctly via loss.backward().
//
// Why forward-mode instead of functional::grad twice?
//   The autograd_engine's matmul vjp_fn uses reduce_broadcast which creates fresh
//   leaf tensors, breaking the create_graph chain at the matmul boundary. The
//   forward-mode approach avoids this by never differentiating w.r.t. x; instead
//   it propagates derivative tensors (all in the weight parameter graph) explicitly.

#pragma once

#include "models/pinn/activations.hpp"
#include "models/pinn/layers.hpp"

#include <cstddef>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

namespace pinn {

enum class Activation { Tanh, ReLU, Sigmoid };

inline std::function<autograd::Tensor(const autograd::Tensor&)>
activation_fn(Activation act)
{
    switch (act) {
        case Activation::Tanh:    return tanh_act;
        case Activation::ReLU:    return relu_act;
        case Activation::Sigmoid: return sigmoid_act;
    }
    throw std::invalid_argument("Unknown activation");
}

//   Derivative bundles                             

/// Result of forward_derivs_1d(): u and its spatial derivatives w.r.t. x.
/// All three are scalar Tensors connected to the network parameter graph.
struct Derivs1D {
    autograd::Tensor u;       // u(x)
    autograd::Tensor du_dx;   // du/dx
    autograd::Tensor d2u_dx2; // d²u/dx²
};

/// Result of forward_derivs_2d(): u and its partial derivatives w.r.t. x and t.
/// All four are scalar Tensors connected to the network parameter graph.
struct Derivs2D {
    autograd::Tensor u;
    autograd::Tensor du_dx;
    autograd::Tensor d2u_dx2;
    autograd::Tensor du_dt;
};

//   NeuralNetwork                                

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<std::size_t> layer_sizes,
                  Activation               act  = Activation::Tanh,
                  unsigned                 seed = 42);

    /// Standard forward pass. x: shape {1, in_features}. Returns shape {1, out_features}.
    autograd::Tensor forward(const autograd::Tensor& input) const;

    //   Forward-mode spatial derivative methods                

    /// Compute u(x), du/dx, d²u/dx² analytically (forward-mode).
    ///
    /// Input must be a 1-D scalar value for a network with input_dim=1.
    /// All returned Tensors are scalar ({1,1}) and are connected to the weight
    /// parameter graph — loss.backward() correctly propagates to all weights.
    ///
    /// Activation: tanh-based formula. For ReLU d²u/dx²≡0 (piecewise linear).
    Derivs1D forward_derivs_1d(double x_val) const;

    /// Compute u, du/dx, d²u/dx², du/dt analytically (forward-mode).
    ///
    /// Network input is [x, t] (input_dim=2). Returns partial derivatives
    /// with respect to each coordinate, all connected to the weight graph.
    Derivs2D forward_derivs_2d(double x_val, double t_val) const;

    //   Parameter management                          

    std::vector<autograd::Tensor*> parameters();
    void zero_grad();

    std::size_t num_layers() const noexcept { return layers_.size(); }
    const std::vector<std::size_t>& layer_sizes() const noexcept { return layer_sizes_; }

    const Linear& layer(std::size_t i) const { return layers_[i]; }
    Linear&       layer(std::size_t i)       { return layers_[i]; }

    Activation activation() const noexcept { return activation_; }

private:
    std::vector<std::size_t> layer_sizes_;
    std::vector<Linear>      layers_;
    Activation               activation_;
    mutable std::mt19937     rng_;
};

}  // namespace pinn
