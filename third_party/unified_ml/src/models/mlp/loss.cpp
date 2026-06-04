#include "models/mlp/loss.hpp"
#include "models/mlp/activation.hpp"
#include "autograd/node.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace mlp {
autograd::Tensor mse_loss(const autograd::Tensor& predictions, const autograd::Tensor& targets) {
    if (predictions.shape() != targets.shape())
        throw std::invalid_argument("mse_loss: shape mismatch");
    autograd::Tensor diff = predictions - targets;
    return autograd::mean(autograd::pow(diff, 2.0));
}
autograd::Tensor bce_loss(const autograd::Tensor& predictions, const autograd::Tensor& targets) {
    if (predictions.shape() != targets.shape())
        throw std::invalid_argument("bce_loss: shape mismatch");
    constexpr double eps = 1e-7;
    autograd::Tensor log_p        = log_tensor(predictions + eps);
    autograd::Tensor one_minus_p  = predictions * (-1.0) + 1.0;
    autograd::Tensor log_1_minus_p= log_tensor(one_minus_p + eps);
    autograd::Tensor one_minus_y  = targets * (-1.0) + 1.0;
    autograd::Tensor inner = autograd::mul(targets, log_p) + autograd::mul(one_minus_y, log_1_minus_p);
    return autograd::mean(inner) * (-1.0);
}

autograd::Tensor bce_with_logits_loss(const autograd::Tensor& logits, const autograd::Tensor& targets) {
    if (logits.shape() != targets.shape())
        throw std::invalid_argument("bce_with_logits_loss: shape mismatch");

    const std::size_t n = logits.numel();
    const std::vector<double>& x = logits.data();
    const std::vector<double>& y = targets.data();
    std::vector<double> loss_terms(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double max_x = std::max(xi, 0.0);
        loss_terms[i] = max_x - xi * yi + std::log1p(std::exp(-std::abs(xi)));
    }

    double mean_loss = 0.0;
    for (double v : loss_terms) mean_loss += v;
    mean_loss /= static_cast<double>(n);

    autograd::Tensor out(mean_loss, false);
    if (!logits.requires_grad()) return out;

    auto out_node = std::make_shared<autograd::Node>(1);
    out_node->is_leaf = false;
    auto x_node = logits.node();
    if (x_node) out_node->inputs.push_back(x_node);

    out_node->backward_fn = [nd=out_node.get(), x_node, x, y, n]() {
        if (!x_node) return;
        const double scale = nd->grad[0] / static_cast<double>(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double sigma = 1.0 / (1.0 + std::exp(-x[i]));
            x_node->grad[i] += scale * (sigma - y[i]);
        }
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}
} // namespace mlp
