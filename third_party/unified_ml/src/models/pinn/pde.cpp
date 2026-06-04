// pde.cpp — PDE residual implementations using forward-mode differentiation.

#include "models/pinn/pde.hpp"

#include <cmath>
#include <stdexcept>

namespace pinn {

constexpr double PI = 3.14159265358979323846;

// Poisson1D            

Poisson1D::Poisson1D(std::function<double(double)> source)
  : source_fn_(std::move(source))
{}

double Poisson1D::source(double x) const
{
  if (source_fn_) return source_fn_(x);
  return PI * PI * std::sin(PI * x);
}

double Poisson1D::exact(double x) const
{
  return std::sin(PI * x);
}

autograd::Tensor Poisson1D::residual(const std::vector<double>& coords,
              NeuralNetwork& net) const
{
  if (coords.size() != 1)
    throw std::invalid_argument("Poisson1D::residual: expected 1 coordinate");

  double x_val = coords[0];

  // Forward-mode: get u, du/dx, d²u/dx² — all connected to weight graph.
  Derivs1D d = net.forward_derivs_1d(x_val);

  // Residual: u_xx + f(x) = 0  (since PDE is -u_xx = f)
  double f_val = source(x_val);
  return d.d2u_dx2 + f_val;
}

// HeatEquation1D           

HeatEquation1D::HeatEquation1D(double alpha)
  : alpha_(alpha)
{}

double HeatEquation1D::exact(double x, double t) const
{
  return std::exp(-alpha_ * PI * PI * t) * std::sin(PI * x);
}

autograd::Tensor HeatEquation1D::residual(const std::vector<double>& coords,
               NeuralNetwork& net) const
{
  if (coords.size() != 2)
    throw std::invalid_argument("HeatEquation1D::residual: expected 2 coordinates [x, t]");

  double x_val = coords[0];
  double t_val = coords[1];

  // Forward-mode: get u, du/dx, d²u/dx², du/dt — all connected to weight graph.
  Derivs2D d = net.forward_derivs_2d(x_val, t_val);

  // Residual: u_t - α u_xx = 0
  return d.du_dt - alpha_ * d.d2u_dx2;
}

}  // namespace pinn
