#pragma once
#include "autograd/autograd.h"
namespace deep_onet {
autograd::Tensor mse_loss(const autograd::Tensor& pred, const autograd::Tensor& target);
autograd::Tensor mae_loss(const autograd::Tensor& pred, const autograd::Tensor& target, double eps = 1e-8);
autograd::Tensor relative_l2_loss(const autograd::Tensor& pred, const autograd::Tensor& target, double eps = 1e-8);
} // namespace deep_onet
