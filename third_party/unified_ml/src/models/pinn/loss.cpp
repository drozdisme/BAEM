// loss.cpp — Loss function implementations.

#include "models/pinn/loss.hpp"

#include <stdexcept>

namespace pinn {

autograd::Tensor squared_error(const autograd::Tensor& pred,
            const autograd::Tensor& target)
{
  autograd::Tensor diff = pred - target;
  return autograd::pow(diff, 2.0);
}

autograd::Tensor mse_scalar_pairs(
  const std::vector<autograd::Tensor>& preds,
  const std::vector<autograd::Tensor>& targets)
{
  if (preds.size() != targets.size())
    throw std::invalid_argument("mse_scalar_pairs: size mismatch");
  if (preds.empty())
    return autograd::Tensor(0.0, false);

  autograd::Tensor loss = squared_error(preds[0], targets[0]);
  for (std::size_t i = 1; i < preds.size(); ++i)
    loss = loss + squared_error(preds[i], targets[i]);

  return loss * (1.0 / static_cast<double>(preds.size()));
}

autograd::Tensor mse(const autograd::Tensor& pred,
        const autograd::Tensor& target)
{
  autograd::Tensor diff = pred - target;
  autograd::Tensor sq = autograd::pow(diff, 2.0);
  return autograd::mean(sq);
}

autograd::Tensor pde_residual_loss(
  const std::vector<autograd::Tensor>& residuals)
{
  if (residuals.empty())
    return autograd::Tensor(0.0, false);

  autograd::Tensor loss = autograd::pow(residuals[0], 2.0);
  for (std::size_t i = 1; i < residuals.size(); ++i)
    loss = loss + autograd::pow(residuals[i], 2.0);

  return loss * (1.0 / static_cast<double>(residuals.size()));
}

}  // namespace pinn
