// neural_network.cpp — NeuralNetwork implementation.
//
// Forward-mode differentiation for PDE residuals
//          
// standard functional::grad breaks for higher-order derivatives through matmul
// because the matmul vjp_fn uses reduce_broadcast which creates fresh leaf
// tensors, severing the computation graph.
//
// The forward-mode approach instead propagates derivative bundles
// (u, du/dx, d²u/dx², ...)
// through each layer analytically, using only standard Tensor ops (mul, matmul,
// add, scalar multiply). Since all ops use weight tensors with requires_grad=true,
// the computation graph w.r.t. parameters is correctly maintained for backward().
//
// Activation formulas used:
// tanh:  σ(z) = tanh(z)
//    σ'(z) = 1 - tanh²(z) (sech²)
//    σ''(z) = -2 tanh(z) (1 - tanh²(z))
// sigmoid: σ(z) = 1/(1+e^{-z})
//    σ'(z) = σ(z)(1-σ(z))
//    σ''(z) = σ'(z)(1-2σ(z))
// relu:  σ'(z) is step function → σ''(z) = 0 (not usable for Laplacians)

#include "models/pinn/neural_network.hpp"

#include <stdexcept>
#include <cmath>

namespace pinn {

//             
//  Construction
//             

NeuralNetwork::NeuralNetwork(std::vector<std::size_t> layer_sizes,
           Activation     act,
           unsigned       seed)
  : layer_sizes_(std::move(layer_sizes))
  , activation_(act)
  , rng_(seed)
{
  if (layer_sizes_.size() < 2)
    throw std::invalid_argument(
    "NeuralNetwork: layer_sizes must have at least 2 entries.");

  layers_.reserve(layer_sizes_.size() - 1);
  for (std::size_t i = 0; i + 1 < layer_sizes_.size(); ++i)
    layers_.emplace_back(layer_sizes_[i], layer_sizes_[i + 1], rng_);
}

//             
//  Standard forward pass
//             

autograd::Tensor NeuralNetwork::forward(const autograd::Tensor& input) const
{
  auto act = activation_fn(activation_);
  autograd::Tensor h = input;

  for (std::size_t i = 0; i < layers_.size(); ++i) {
    h = layers_[i].forward(h);
    if (i + 1 < layers_.size())
    h = act(h);
  }
  return h;
}

//             
//  File-local helpers: activation derivatives
//             

namespace {

/// Apply activation and compute σ(z), σ'(z), σ''(z) as Tensors.
/// All returned tensors are connected to the graph through z (which depends on weights).
struct ActivationOutput {
  autograd::Tensor a;  // σ(z)
  autograd::Tensor sp; // σ'(z)
  autograd::Tensor spp;  // σ''(z)
};

ActivationOutput apply_activation_with_derivs(const autograd::Tensor& z, Activation act)
{
  if (act == Activation::Tanh) {
    autograd::Tensor tanh_z = tanh_act(z);        // tanh(z)
    autograd::Tensor tanh_sq = autograd::mul(tanh_z, tanh_z);   // tanh²(z)
    autograd::Tensor sp  = 1.0 - tanh_sq;         // sech²(z)
    autograd::Tensor spp = (-2.0) * autograd::mul(tanh_z, sp);  // -2 tanh·sech²
    return {tanh_z, sp, spp};
  }
  else if (act == Activation::Sigmoid) {
    autograd::Tensor sig  = sigmoid_act(z);        // σ(z)
    autograd::Tensor omos = 1.0 - sig;           // 1 - σ(z)
    autograd::Tensor sp = autograd::mul(sig, omos);      // σ(1-σ)
    autograd::Tensor spp  = autograd::mul(sp, 1.0 - 2.0 * sig); // σ'(1-2σ)
    return {sig, sp, spp};
  }
  else { // ReLU — second derivative is 0
    autograd::Tensor a  = relu_act(z);
    // σ'(z): step(z) = relu(z) / z ≈ (z > 0 ? 1 : 0), approximated
    // For PDE residuals with ReLU, d²u/dx²≡0, so spp = 0 is correct.
    // sp: create a zero tensor with the same shape, no grad (it IS zero)
    autograd::Tensor sp  = autograd::Tensor(
    std::vector<double>(z.numel(), 0.0), z.shape(), false);
    autograd::Tensor spp = autograd::Tensor(
    std::vector<double>(z.numel(), 0.0), z.shape(), false);
    return {a, sp, spp};
  }
}

/// One hidden layer forward with derivative propagation.
/// state_u: current activation  shape {1, width}
/// state_d1: current du/d_coord   shape {1, width}
/// state_d2: current d²u/d_coord² shape {1, width}
/// layer: the Linear layer
/// act: activation type
/// Returns updated (u, d1, d2) after this layer.
std::tuple<autograd::Tensor, autograd::Tensor, autograd::Tensor>
layer_forward_derivs(
  const autograd::Tensor& state_u,
  const autograd::Tensor& state_d1,
  const autograd::Tensor& state_d2,
  const Linear& layer,
  Activation act)
{
  // Linear: z = u @ W + b
  autograd::Tensor z = layer.forward(state_u);
  // Chain rule for first derivative: dz/d_coord = du/d_coord @ W
  autograd::Tensor dz  = autograd::matmul(state_d1, layer.weight());
  // Chain rule for second derivative:  d²z/d_coord² = d²u/d_coord² @ W
  autograd::Tensor d2z = autograd::matmul(state_d2, layer.weight());

  // Activation and its derivatives at z (all connected to weight graph).
  auto [a, sp, spp] = apply_activation_with_derivs(z, act);

  // First derivative through activation: da/d_coord = σ'(z) * dz/d_coord
  autograd::Tensor da = autograd::mul(sp, dz);

  // Second derivative: d²a/d_coord² = σ''(z)*(dz/d_coord)² + σ'(z)*d²z/d_coord²
  autograd::Tensor d2a = autograd::mul(spp, autograd::mul(dz, dz))
         + autograd::mul(sp, d2z);

  return {a, da, d2a};
}

} // anonymous namespace

//             
//  forward_derivs_1d
//             

Derivs1D NeuralNetwork::forward_derivs_1d(double x_val) const
{
  if (layer_sizes_.front() != 1)
    throw std::invalid_argument(
    "forward_derivs_1d: network input_dim must be 1");

  const std::size_t n_layers = layers_.size();  // includes output layer

  // Initial state: u = x (no grad on x itself — we use forward-mode),
  // du/dx = 1, d²u/dx² = 0.
  // Shape: {1, 1} matching the first layer's input.
  autograd::Tensor u ({x_val}, {1, 1}, false);
  autograd::Tensor d1  ({1.0}, {1, 1}, false);
  autograd::Tensor d2  ({0.0}, {1, 1}, false);

  // Hidden layers (all but the last) — with activation.
  for (std::size_t i = 0; i + 1 < n_layers; ++i) {
    auto [new_u, new_d1, new_d2] =
    layer_forward_derivs(u, d1, d2, layers_[i], activation_);
    u  = std::move(new_u);
    d1 = std::move(new_d1);
    d2 = std::move(new_d2);
  }

  // Output layer — linear only (no activation).
  {
    const Linear& out_layer = layers_[n_layers - 1];
    autograd::Tensor z = out_layer.forward(u);
    autograd::Tensor dz  = autograd::matmul(d1, out_layer.weight());
    autograd::Tensor d2z = autograd::matmul(d2, out_layer.weight());
    u  = z;
    d1 = dz;
    d2 = d2z;
  }

  // Convert to scalars via sum (output is {1,1}).
  return {autograd::sum(u), autograd::sum(d1), autograd::sum(d2)};
}

//             
//  forward_derivs_2d  (for heat equation: input is [x, t])
//             

Derivs2D NeuralNetwork::forward_derivs_2d(double x_val, double t_val) const
{
  if (layer_sizes_.front() != 2)
    throw std::invalid_argument(
    "forward_derivs_2d: network input_dim must be 2");

  const std::size_t n_layers = layers_.size();

  // Initial state for input [x, t]:
  // du/dx initial = [1, 0] (x-component of gradient)
  // d²u/dx² init  = [0, 0]
  // du/dt initial = [0, 1] (t-component of gradient)
  autograd::Tensor u  ({x_val, t_val}, {1, 2}, false);
  autograd::Tensor dx1  ({1.0, 0.0},   {1, 2}, false);  // du/dx seeding
  autograd::Tensor dx2  ({0.0, 0.0},   {1, 2}, false);  // d²u/dx²
  autograd::Tensor dt1  ({0.0, 1.0},   {1, 2}, false);  // du/dt seeding

  // Hidden layers with activation.
  for (std::size_t i = 0; i + 1 < n_layers; ++i) {
    // Compute linear part once, then reuse for x/t derivative paths.
    const Linear& layer = layers_[i];
    autograd::Tensor z = layer.forward(u);
    autograd::Tensor dz_dx = autograd::matmul(dx1, layer.weight());
    autograd::Tensor d2z_dx = autograd::matmul(dx2, layer.weight());
    autograd::Tensor dz_dt = autograd::matmul(dt1, layer.weight());

    auto [a, sp, spp] = apply_activation_with_derivs(z, activation_);
    autograd::Tensor new_dx1 = autograd::mul(sp, dz_dx);
    autograd::Tensor new_dx2 = autograd::mul(spp, autograd::mul(dz_dx, dz_dx))
                             + autograd::mul(sp, d2z_dx);
    autograd::Tensor new_dt1 = autograd::mul(sp, dz_dt);

    u = std::move(a);
    dx1 = std::move(new_dx1);
    dx2 = std::move(new_dx2);
    dt1 = std::move(new_dt1);
  }

  // Output layer — linear only.
  {
    const Linear& out_layer = layers_[n_layers - 1];
    autograd::Tensor z_out = out_layer.forward(u);
    autograd::Tensor dz_dx  = autograd::matmul(dx1, out_layer.weight());
    autograd::Tensor d2z_dx = autograd::matmul(dx2, out_layer.weight());
    autograd::Tensor dz_dt  = autograd::matmul(dt1, out_layer.weight());
    u = z_out;
    dx1 = dz_dx;
    dx2 = d2z_dx;
    dt1 = dz_dt;
  }

  return {autograd::sum(u),
    autograd::sum(dx1),
    autograd::sum(dx2),
    autograd::sum(dt1)};
}

//             
//  Parameter management
//             

std::vector<autograd::Tensor*> NeuralNetwork::parameters()
{
  std::vector<autograd::Tensor*> params;
  for (auto& layer : layers_) {
    auto lp = layer.parameters();
    params.insert(params.end(), lp.begin(), lp.end());
  }
  return params;
}

void NeuralNetwork::zero_grad()
{
  for (auto& layer : layers_)
    layer.zero_grad();
}

}  // namespace pinn
