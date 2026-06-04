#pragma once
#include "models/deep_onet/branch_net.hpp"
#include "models/deep_onet/trunk_net.hpp"
#include "autograd/autograd.h"
#include "ucao/engine_policy.hpp"
#include <cstddef>
#include <memory>
#include <vector>
namespace deep_onet {
class DeepONet : public ucao::engine::PolicyBound<ucao::engine::ModelFamily::DeepONet> {
public:
    DeepONet(std::size_t branch_input_dim, const std::vector<std::size_t>& branch_hidden_dims,
             std::size_t trunk_input_dim,  const std::vector<std::size_t>& trunk_hidden_dims,
             std::size_t latent_dim,
             Activation branch_act = Activation::ReLU,
             Activation trunk_act  = Activation::Tanh);
    DeepONet(const DeepONet&) = delete;
    DeepONet& operator=(const DeepONet&) = delete;
    DeepONet(DeepONet&&) = default;
    DeepONet& operator=(DeepONet&&) = default;
    autograd::Tensor forward(const autograd::Tensor& u_batch, const autograd::Tensor& y_batch) const;
    std::vector<autograd::Tensor*> parameters();
    void zero_grad();
    BranchNet& branch()             { return *branch_; }
    TrunkNet&  trunk()              { return *trunk_;  }
    const BranchNet& branch() const { return *branch_; }
    const TrunkNet&  trunk()  const { return *trunk_;  }
    std::size_t latent_dim() const  { return latent_dim_; }
private:
    std::size_t latent_dim_;
    std::unique_ptr<BranchNet> branch_;
    std::unique_ptr<TrunkNet>  trunk_;
    std::unique_ptr<autograd::Tensor> branch_latent_gain_;
    std::unique_ptr<autograd::Tensor> branch_latent_bias_;
    std::unique_ptr<autograd::Tensor> branch_adapter_mix_;
    std::unique_ptr<autograd::Tensor> trunk_latent_gain_;
    std::unique_ptr<autograd::Tensor> trunk_latent_bias_;
    std::unique_ptr<autograd::Tensor> trunk_adapter_mix_;
    std::unique_ptr<autograd::Tensor> output_scale_;
    std::unique_ptr<autograd::Tensor> output_bias_;
};
} // namespace deep_onet
