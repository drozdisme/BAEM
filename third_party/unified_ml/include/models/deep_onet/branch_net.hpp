#pragma once
#include "models/deep_onet/layers.hpp"
#include "models/deep_onet/activations.hpp"
#include <cstddef>
#include <vector>
namespace deep_onet {
class BranchNet {
public:
    BranchNet(std::size_t input_dim, const std::vector<std::size_t>& hidden_dims,
              std::size_t output_dim, Activation act = Activation::ReLU);
    BranchNet(const BranchNet&) = delete;
    BranchNet& operator=(const BranchNet&) = delete;
    BranchNet(BranchNet&&) = default;
    BranchNet& operator=(BranchNet&&) = default;
    autograd::Tensor forward(const autograd::Tensor& u) const;
    std::vector<autograd::Tensor*> parameters();
    void zero_grad();
    std::size_t input_dim()  const { return input_dim_; }
    std::size_t output_dim() const { return output_dim_; }
private:
    std::size_t input_dim_, output_dim_;
    Activation  activation_;
    std::vector<Linear> layers_;
};
} // namespace deep_onet
