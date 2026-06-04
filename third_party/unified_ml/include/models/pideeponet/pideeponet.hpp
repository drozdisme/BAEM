#pragma once

#include "autograd/autograd.h"
#include "models/deep_onet/model.hpp"
#include "ucao/engine_policy.hpp"

#include <functional>
#include <string>
#include <vector>

namespace pideeponet {

using ResidualFn = std::function<autograd::Tensor(const autograd::Tensor&, const autograd::Tensor&)>;

struct LossBreakdown {
    autograd::Tensor total;
    autograd::Tensor data;
    autograd::Tensor physics;
};

struct PIDeepONetConfig {
    std::size_t branch_input_dim;
    std::vector<std::size_t> branch_hidden_dims;
    std::size_t trunk_input_dim;
    std::vector<std::size_t> trunk_hidden_dims;
    std::size_t latent_dim;
    deep_onet::Activation branch_act = deep_onet::Activation::ReLU;
    deep_onet::Activation trunk_act  = deep_onet::Activation::Tanh;
};

class PIDeepONet : public ucao::engine::PolicyBound<ucao::engine::ModelFamily::PIDeepONet> {
public:
    // Experimental contract:
    // - forward/loss operate on rank-2 batches only
    // - residual_fn must preserve autograd semantics if used for training
    // - save/load snapshots preserve fitted parameter state, but not a full training-runtime contract
    explicit PIDeepONet(const PIDeepONetConfig& config);
    PIDeepONet(std::size_t branch_input_dim,
               const std::vector<std::size_t>& branch_hidden_dims,
               std::size_t trunk_input_dim,
               const std::vector<std::size_t>& trunk_hidden_dims,
               std::size_t latent_dim,
               deep_onet::Activation branch_act = deep_onet::Activation::ReLU,
               deep_onet::Activation trunk_act  = deep_onet::Activation::Tanh);

    autograd::Tensor forward(const autograd::Tensor& u_batch,
                             const autograd::Tensor& y_batch) const;

    LossBreakdown loss(const autograd::Tensor& u_batch,
                       const autograd::Tensor& y_batch,
                       const autograd::Tensor& target,
                       const ResidualFn& residual_fn,
                       double physics_weight = 1.0,
                       double data_weight = 1.0) const;

    std::vector<autograd::Tensor*> parameters();
    void zero_grad();

    const PIDeepONetConfig& config() const noexcept { return config_; }
    void save(const std::string& filepath) const;
    static PIDeepONet load(const std::string& filepath);

    static constexpr bool has_stable_serialization_contract() noexcept { return false; }

private:
    PIDeepONetConfig config_;
    deep_onet::DeepONet model_;
};

}  // namespace pideeponet
