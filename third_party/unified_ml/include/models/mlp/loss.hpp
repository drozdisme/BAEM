#pragma once
#include "autograd/tensor.h"
namespace mlp {
autograd::Tensor mse_loss(const autograd::Tensor& predictions, const autograd::Tensor& targets);
autograd::Tensor bce_loss(const autograd::Tensor& predictions, const autograd::Tensor& targets);
autograd::Tensor bce_with_logits_loss(const autograd::Tensor& logits, const autograd::Tensor& targets);
} // namespace mlp
