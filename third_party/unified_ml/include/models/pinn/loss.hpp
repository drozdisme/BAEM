// loss.h — Loss function building blocks for Physics-Informed Neural Networks.
//
// Functions:
//   mse(pred, target)          — Mean Squared Error between two scalar tensors.
//   mse_batch(preds, targets)  — MSE over a collection of scalar pairs.
//
// The PINN training loss is composed of three terms:
//   L = λ_pde · L_pde  +  λ_bc · L_bc  +  λ_data · L_data
//
// Each term is computed externally in the training loop (see pinn_model.h)
// and accumulated as a single scalar Tensor before calling backward().

#pragma once

#include "autograd/tensor.h"

#include <vector>

namespace pinn {

/// Squared error between two scalar Tensors: (pred - target)^2.
/// Returns a scalar Tensor that participates in the computation graph.
autograd::Tensor squared_error(const autograd::Tensor& pred,
                                const autograd::Tensor& target);

/// MSE over N scalar pairs: (1/N) Σ (pred_i - target_i)^2.
/// Each pred_i and target_i are scalar Tensors.
autograd::Tensor mse_scalar_pairs(
    const std::vector<autograd::Tensor>& preds,
    const std::vector<autograd::Tensor>& targets);

/// MSE between two identically-shaped Tensors (element-wise).
autograd::Tensor mse(const autograd::Tensor& pred,
                      const autograd::Tensor& target);

/// Sum of squared PDE residuals over a list of scalar residual Tensors.
autograd::Tensor pde_residual_loss(
    const std::vector<autograd::Tensor>& residuals);

}  // namespace pinn
