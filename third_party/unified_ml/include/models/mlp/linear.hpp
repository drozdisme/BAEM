#pragma once
#include "models/mlp/layer.hpp"
#include <cstddef>

namespace mlp {
class Linear final : public Layer {
public:
    Linear(std::size_t in_features, std::size_t out_features, bool use_bias = true);
    autograd::Tensor forward(const autograd::Tensor& input) override;
    std::vector<autograd::Tensor*> parameters() override;
    autograd::Tensor&       weight() noexcept { return weight_; }
    const autograd::Tensor& weight() const noexcept { return weight_; }
    autograd::Tensor&       bias()   noexcept { return bias_; }
    const autograd::Tensor& bias()   const noexcept { return bias_; }
    std::size_t in_features()  const noexcept { return in_features_; }
    std::size_t out_features() const noexcept { return out_features_; }
    bool        has_bias()     const noexcept { return use_bias_; }
private:
    std::size_t      in_features_, out_features_;
    bool             use_bias_;
    autograd::Tensor weight_, bias_;
};
} // namespace mlp
