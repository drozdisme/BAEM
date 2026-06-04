// pde.h — Abstract PDE interface and concrete implementations.
//
// PDE residuals are computed using the network's forward-mode derivative
// methods (forward_derivs_1d / forward_derivs_2d), which propagate spatial
// derivative bundles analytically through the network layers.
//
// All returned residual Tensors are scalar ({1,1}) and are connected to the
// weight parameter graph, so loss.backward() propagates gradients correctly.

#pragma once

#include "models/pinn/neural_network.hpp"

#include <cmath>
#include <functional>

namespace pinn {

//   Abstract base                                

class PDE {
public:
    virtual ~PDE() = default;

    /// Compute the PDE residual at a single interior point.
    /// @param coords   Coordinate values (e.g. x, or {x, t}).
    /// @param net      The network (forward-mode derivatives computed internally).
    /// @return         Scalar Tensor; residual should be 0 at the true solution.
    virtual autograd::Tensor residual(const std::vector<double>& coords,
                                       NeuralNetwork& net) const = 0;

    /// Input dimensionality (1 for 1-D PDEs, 2 for (x,t), etc.).
    virtual std::size_t input_dim() const noexcept = 0;
};

//   1D Poisson  -d²u/dx² = f(x)                        

class Poisson1D : public PDE {
public:
    /// @param source  Optional override for f(x); default = π²sin(πx).
    explicit Poisson1D(std::function<double(double)> source = {});

    autograd::Tensor residual(const std::vector<double>& coords,
                               NeuralNetwork& net) const override;

    std::size_t input_dim() const noexcept override { return 1; }

    double source(double x) const;
    double exact (double x) const;

private:
    std::function<double(double)> source_fn_;
};

//   1D Heat equation  u_t = α u_xx                      

class HeatEquation1D : public PDE {
public:
    explicit HeatEquation1D(double alpha = 0.01);

    autograd::Tensor residual(const std::vector<double>& coords,
                               NeuralNetwork& net) const override;

    std::size_t input_dim() const noexcept override { return 2; }

    double exact(double x, double t) const;
    double alpha() const noexcept { return alpha_; }

private:
    double alpha_;
};

}  // namespace pinn
