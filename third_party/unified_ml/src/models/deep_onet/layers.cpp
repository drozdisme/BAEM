#include "models/deep_onet/layers.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace deep_onet {
Linear::Linear(std::size_t in_f, std::size_t out_f, bool use_bias, unsigned layer_index)
    : in_features_(in_f), out_features_(out_f), use_bias_(use_bias)
{
    if (in_f == 0 || out_f == 0) throw std::invalid_argument("Linear: dimensions must be > 0");
    const double bound = std::sqrt(6.0 / static_cast<double>(in_f));
    const unsigned seed = static_cast<unsigned>((in_f * 1000003u) ^ (out_f * 999983u) ^ (layer_index * 2654435761u));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-bound, bound);
    const std::size_t w_numel = out_f * in_f;
    std::vector<double> w_data(w_numel);
    for (auto& v : w_data) v = dist(rng);
    weight_ = std::make_unique<autograd::Tensor>(std::move(w_data), std::vector<std::size_t>{out_f, in_f}, true);
    if (use_bias_) {
        std::vector<double> b_data(out_f, 0.0);
        bias_ = std::make_unique<autograd::Tensor>(std::move(b_data), std::vector<std::size_t>{out_f}, true);
    }
}
autograd::Tensor Linear::forward(const autograd::Tensor& x) const {
    if (x.shape().back() != in_features_)
        throw std::invalid_argument("Linear::forward: input last-dim mismatch");
    autograd::Tensor wt = autograd::transpose(*weight_);
    autograd::Tensor y  = autograd::matmul(x, wt);
    if (use_bias_) y = y + *bias_;
    return y;
}
std::vector<autograd::Tensor*> Linear::parameters() {
    std::vector<autograd::Tensor*> p;
    p.push_back(weight_.get());
    if (use_bias_) p.push_back(bias_.get());
    return p;
}
void Linear::zero_grad() {
    weight_->zero_grad();
    if (use_bias_) bias_->zero_grad();
}
} // namespace deep_onet
