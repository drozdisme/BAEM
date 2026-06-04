// pinn_fast.hpp — Zero-overhead analytical gradient PINN engine.
//
// ARCHITECTURE TARGET: general fully-connected tanh network, optimized for
// the 8→64→32→1 configuration (or any similar small-to-medium depth).
//
// WHY THIS EXISTS
//         
// The original NeuralNetwork + autograd dynamic graph approach:
//   • Allocates ~30 shared_ptr<Node> + vector<double> PER collocation point
//   • Builds a new graph from scratch on every forward pass
//   • Topological sort (unordered_set + DFS) every backward call
//   • 200x slower than Python numpy/autograd due to allocator pressure
//
// This implementation:
//   • ZERO heap allocation per training step (pre-allocated buffers)
//   • ZERO shared_ptr ref-counting overhead
//   • Manually inlines the full forward + analytical backward pass
//   • Works on raw aligned double[] arrays throughout
//   • Uses #pragma omp simd on all inner loops
//
// ANALYTICAL GRADIENT DERIVATION
//                 
// For a tanh network y = f_L ∘ ... ∘ f_1(x) with layers:
//   z_ℓ = W_ℓ @ a_{ℓ-1} + b_ℓ
//   a_ℓ = tanh(z_ℓ)
//   s_ℓ = 1 - a_ℓ²          (sech², first-order activation derivative)
//   p_ℓ = -2 a_ℓ s_ℓ         (second-order activation derivative)
//   q_ℓ = (-2 + 6 a_ℓ²) s_ℓ  (third-order, needed for d²u/dx² gradient)
//
// Forward-mode derivatives through the network (1D input x):
//   dz_ℓ  = W_ℓ @ da_{ℓ-1}           (chain rule for x-derivative of pre-activation)
//   da_ℓ  = s_ℓ ⊙ dz_ℓ               (x-derivative of post-activation)
//   d2z_ℓ = W_ℓ @ d2a_{ℓ-1}
//   d2a_ℓ = s_ℓ ⊙ d2z_ℓ + p_ℓ ⊙ dz_ℓ² (x-second-derivative)
//
// Output layer (linear, no activation):
//   u    = W_out @ a_{L-1} + b_out
//   u'   = W_out @ da_{L-1}
//   u''  = W_out @ d2a_{L-1}
//
// Reverse-mode gradient of loss through u'' = d²u/dx²
//                            
// Three sensitivity "streams" propagate backwards:
//   δa_ℓ   = ∂L/∂a_ℓ   (sensitivity to activation value)
//   δda_ℓ  = ∂L/∂(da_ℓ)  (sensitivity to first x-derivative)
//   δd2a_ℓ = ∂L/∂(d2a_ℓ) (sensitivity to second x-derivative)
//
// Seed (from PDE residual r = u'' + f, loss = r²):
//   δd2a_{L-1} = ∂L/∂r * W_out[0,:]  (upstream × output row)
//   δda_{L-1}  = 0   (PDE residual doesn't see du/dx directly)
//   δa_{L-1}   = 0
//
// At each hidden layer ℓ (going backwards):
//   δd2z = δd2a ⊙ s_ℓ
//   δdz  = δda ⊙ s_ℓ  +  2 * δd2a ⊙ p_ℓ ⊙ dz_ℓ
//   δz   = δa ⊙ s_ℓ   +  δda ⊙ p_ℓ ⊙ dz_ℓ  +  δd2a ⊙ (p_ℓ ⊙ d2z_ℓ + q_ℓ ⊙ dz_ℓ²)
//
//   ∂L/∂b_ℓ = δz               (gradient w.r.t. bias)
//   ∂L/∂W_ℓ = δz ⊗ a_{ℓ-1}^T  +  δdz ⊗ da_{ℓ-1}^T  +  δd2z ⊗ d2a_{ℓ-1}^T
//
//   δa_{ℓ-1}   = W_ℓ^T @ δz
//   δda_{ℓ-1}  = W_ℓ^T @ δdz
//   δd2a_{ℓ-1} = W_ℓ^T @ δd2z

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>
#include "core/compat.hpp"

namespace pinn {

//   AlignedBuffer                                
// Stack-like heap buffer, 64-byte aligned, zero-overhead access.
struct AlignedBuffer {
    double* ptr = nullptr;
    std::size_t cap = 0;

    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t n) : cap(n) {
        std::size_t bytes = ((n * sizeof(double) + 63) / 64) * 64;
        ptr = static_cast<double*>(hpc_aligned_alloc(64, bytes));
        if (!ptr) throw std::bad_alloc();
        std::memset(ptr, 0, bytes);
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), cap(o.cap) { o.ptr=nullptr; o.cap=0; }
    ~AlignedBuffer() { hpc_aligned_free(ptr); }

    void zero() noexcept { if(ptr) std::memset(ptr, 0, cap * sizeof(double)); }
    double& operator[](std::size_t i) noexcept { return ptr[i]; }
    const double& operator[](std::size_t i) const noexcept { return ptr[i]; }
};

//   PinnFastLayer                                
// One fully-connected layer: y = x @ W + b  (W stored row-major)
// Buffers for activations/derivatives pre-allocated — no per-step alloc.
struct PinnFastLayer {
    std::size_t in_dim, out_dim;

    // Parameters (learnable)
    AlignedBuffer W;    // shape [out_dim × in_dim], row-major
    AlignedBuffer b;    // shape [out_dim]

    // Adam state
    AlignedBuffer mW, vW;   // first/second moments for W
    AlignedBuffer mb, vb;   // first/second moments for b
    AlignedBuffer gW, gb;   // gradient accumulators

    // Forward-pass caches (sized for one point at a time)
    AlignedBuffer z;       // pre-activation:  [out_dim]
    AlignedBuffer a;       // activation:      [out_dim]
    AlignedBuffer s;       // sech²(z):        [out_dim]  (first act deriv)
    AlignedBuffer p;       // -2a·s:           [out_dim]  (second act deriv)
    AlignedBuffer q;       // (-2+6a²)·s:      [out_dim]  (third act deriv)
    AlignedBuffer dz;      // dz/dx:           [out_dim]
    AlignedBuffer da;      // da/dx:           [out_dim]
    AlignedBuffer d2z;     // d²z/dx²:         [out_dim]
    AlignedBuffer d2a;     // d²a/dx²:         [out_dim]

    // Backward sensitivity streams
    AlignedBuffer delta_a;   // ∂L/∂a
    AlignedBuffer delta_da;  // ∂L/∂(da)
    AlignedBuffer delta_d2a; // ∂L/∂(d²a)
    AlignedBuffer delta_z;   // combined δz
    AlignedBuffer delta_dz;
    AlignedBuffer delta_d2z;

    PinnFastLayer() = default;
    PinnFastLayer(std::size_t in, std::size_t out, unsigned& seed);

    // Zero gradient accumulators (called before each batch/step)
    void zero_grad() noexcept;
    // Adam update (called after backward)
    void adam_step(double lr, double beta1, double beta2, double eps, int t) noexcept;
};

//   PinnFastNet                                 
// Drop-in replacement for pinn::NeuralNetwork with:
//   • No dynamic graph construction (zero shared_ptr/Node overhead)
//   • All intermediate buffers pre-allocated once at construction
//   • Analytical backward pass for PINN Poisson residual loss
//   • Integrated Adam optimizer (no separate optimizer object needed)
//
// API: compatible with pinn::PINNModel usage patterns.
class PinnFastNet {
public:
    // layer_sizes: e.g. {1, 64, 32, 1} for 1→64→32→1
    // All layers use tanh activation except the last (linear output).
    explicit PinnFastNet(const std::vector<std::size_t>& layer_sizes,
                         unsigned seed = 42,
                         double lr     = 1e-3,
                         double beta1  = 0.9,
                         double beta2  = 0.999,
                         double eps    = 1e-8);

    //   Inference                                
    // Evaluate u(x) for a vector of input coordinates.
    // x_vals: flat array of n_inputs doubles per point × n_points.
    std::vector<double> predict(const std::vector<double>& x_flat) const;

    // Single-point forward: returns u(x)  (no grad tracking)
    double forward_scalar(const double* x) const noexcept;

    //   PINN training step                            
    // Trains one step on the provided collocation + boundary + data points.
    //
    // coll_x:    flattened collocation coordinates [n_coll × input_dim]
    // f_vals:    PDE forcing values at each coll point [n_coll]
    // bc_x:      flattened BC coordinates [n_bc × input_dim]
    // bc_u:      BC values [n_bc]
    //
    // Returns total loss for this step.
    double train_step_poisson1d(
        const std::vector<double>& coll_x,    // [n_coll]
        const std::vector<double>& f_vals,    // [n_coll]
        const std::vector<double>& bc_x,      // [n_bc]
        const std::vector<double>& bc_u,      // [n_bc]
        double w_pde = 1.0,
        double w_bc  = 1.0);

    // General Poisson-like: residual = Laplacian(u) + f(x)
    // Handles any 1D input dimension.
    double train_step_pde_residual(
        const std::vector<double>& coll_x,
        std::function<double(const double*, double)> residual_fn,
        const std::vector<double>& bc_x,
        const std::vector<double>& bc_u,
        double w_pde = 1.0,
        double w_bc  = 1.0);

    std::size_t input_dim()  const noexcept { return layers_.empty() ? 0 : layers_[0].in_dim; }
    std::size_t output_dim() const noexcept { return layers_.empty() ? 0 : layers_.back().out_dim; }
    std::size_t n_layers()   const noexcept { return layers_.size(); }

    // Zero all gradient buffers
    void zero_grad() noexcept;
    // Perform Adam step on all layers
    void adam_step() noexcept;
    // Increment Adam time step
    void tick() noexcept { ++adam_t_; }

    int adam_t() const noexcept { return adam_t_; }

private:
    std::vector<PinnFastLayer> layers_;
    std::size_t input_dim_;
    double lr_, beta1_, beta2_, eps_;
    int adam_t_ = 0;

    // Forward pass (fills all cache buffers in each layer)
    // x_in: input vector of length input_dim
    // Returns output (scalar for u)
    double forward_and_cache(const double* x_in) noexcept;

    // Forward pass with derivative propagation (1D input x scalar)
    // Fills z,a,s,p,q,dz,da,d2z,d2a in each layer
    // Returns {u, du/dx, d²u/dx²}
    struct Derivs { double u, du, d2u; };
    Derivs forward_derivs_1d(double x_val) noexcept;

    // Backward pass for PDE residual: r = d²u/dx² + f
    // Accumulates dL/dW, dL/db into gW, gb of each layer.
    // dL_dr: upstream gradient (= 2r/N for MSE)
    void backward_pde_residual(double dL_dr) noexcept;

    // Backward pass for BC/data loss: loss = (u - u_ref)²
    // Accumulates gradients into gW, gb.
    // dL_du: upstream gradient (= 2*(u - u_ref)/N)
    void backward_u(double dL_du) noexcept;
};

} // namespace pinn
