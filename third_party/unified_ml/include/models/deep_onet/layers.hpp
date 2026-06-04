#pragma once
#include "autograd/autograd.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace deep_onet {
class Linear {
public:
    Linear(std::size_t in_features, std::size_t out_features, bool use_bias = true, unsigned layer_index = 0);
    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;
    Linear(Linear&&) = default;
    Linear& operator=(Linear&&) = default;
    autograd::Tensor forward(const autograd::Tensor& x) const;
    autograd::Tensor& weight()             { return *weight_; }
    const autograd::Tensor& weight() const { return *weight_; }
    autograd::Tensor& bias()               { return *bias_; }
    const autograd::Tensor& bias()   const { return *bias_; }
    bool        has_bias()      const { return use_bias_; }
    std::size_t in_features()   const { return in_features_; }
    std::size_t out_features()  const { return out_features_; }
    std::vector<autograd::Tensor*> parameters();
    void zero_grad();
private:
    std::size_t in_features_, out_features_;
    bool        use_bias_;
    std::unique_ptr<autograd::Tensor> weight_, bias_;
};
} // namespace deep_onet
