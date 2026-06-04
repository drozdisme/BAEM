#include "models/deep_onet/loss.hpp"
#include <cmath>
#include <stdexcept>
namespace deep_onet {
autograd::Tensor mse_loss(const autograd::Tensor& pred, const autograd::Tensor& target) {
    if (pred.numel() != target.numel()) throw std::invalid_argument("mse_loss: size mismatch");
    autograd::Tensor diff = pred - target;
    return autograd::mean(diff * diff);
}
autograd::Tensor mae_loss(const autograd::Tensor& pred, const autograd::Tensor& target, double eps) {
    if (pred.numel() != target.numel()) throw std::invalid_argument("mae_loss: size mismatch");
    autograd::Tensor diff = pred - target;
    return autograd::mean(autograd::pow(diff * diff + eps, 0.5));
}
autograd::Tensor relative_l2_loss(const autograd::Tensor& pred, const autograd::Tensor& target, double eps) {
    if (pred.numel() != target.numel()) throw std::invalid_argument("relative_l2_loss: size mismatch");
    autograd::Tensor diff = pred - target;
    autograd::Tensor num  = autograd::sum(diff * diff);
    double denom = eps;
    for (double v : target.data()) denom += v * v;
    return num / denom;
}
} // namespace deep_onet
