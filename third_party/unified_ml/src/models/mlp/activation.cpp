#include "models/mlp/activation.hpp"
#include "autograd/node.h"
#include <cmath>
#include <memory>
#include <stdexcept>

namespace mlp {
autograd::Tensor exp_tensor(const autograd::Tensor& a) {
    const std::size_t n = a.numel();
    std::vector<double> out_data(n);
    for (std::size_t i = 0; i < n; ++i) out_data[i] = std::exp(a.data()[i]);
    autograd::Tensor out(out_data, a.shape(), false);
    if (!a.requires_grad()) return out;
    auto out_node = std::make_shared<autograd::Node>(n);
    out_node->is_leaf = false;
    auto a_node = a.node();
    if (a_node) out_node->inputs.push_back(a_node);
    auto exp_vals = out_data;
    out_node->backward_fn = [nd=out_node.get(), a_node, exp_vals]() {
        if (!a_node) return;
        for (std::size_t i = 0; i < a_node->grad.size(); ++i)
            a_node->grad[i] += nd->grad[i] * exp_vals[i];
    };
    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}
autograd::Tensor log_tensor(const autograd::Tensor& a) {
    const std::size_t n = a.numel();
    std::vector<double> out_data(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (a.data()[i] <= 0.0) throw std::domain_error("log_tensor: all inputs must be strictly positive");
        out_data[i] = std::log(a.data()[i]);
    }
    autograd::Tensor out(out_data, a.shape(), false);
    if (!a.requires_grad()) return out;
    auto out_node = std::make_shared<autograd::Node>(n);
    out_node->is_leaf = false;
    auto a_node = a.node();
    auto a_data = a.data();
    if (a_node) out_node->inputs.push_back(a_node);
    out_node->backward_fn = [nd=out_node.get(), a_node, a_data]() {
        if (!a_node) return;
        for (std::size_t i = 0; i < a_node->grad.size(); ++i)
            a_node->grad[i] += nd->grad[i] / a_data[i];
    };
    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}
autograd::Tensor sigmoid_fn(const autograd::Tensor& a) {
    autograd::Tensor neg_a = -a;
    autograd::Tensor e     = exp_tensor(neg_a);
    autograd::Tensor denom = e + 1.0;
    return autograd::pow(denom, -1.0);
}
autograd::Tensor tanh_fn(const autograd::Tensor& a) {
    autograd::Tensor two_a = a * 2.0;
    autograd::Tensor sig   = sigmoid_fn(two_a);
    return sig * 2.0 - 1.0;
}
} // namespace mlp
