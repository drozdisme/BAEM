#pragma once
#include "models/mlp/sequential.hpp"
#include "core/optimizers.hpp"
#include "ucao/engine_policy.hpp"
#include <functional>
#include <memory>
#include <variant>
namespace mlp {
using SGD = core::SGD;
using Adam = core::Adam;
using RMSProp = core::RMSProp;
using NAdam = core::NAdam;

enum class OptimizerType {
    Auto,
    SGD,
    Adam,
    RMSProp,
    NAdam,
};

class Model {
public:
    Model(std::unique_ptr<Sequential> network, double learning_rate);
    Model(std::unique_ptr<Sequential> network, double learning_rate, OptimizerType optimizer_type);
    autograd::Tensor forward(const autograd::Tensor& input);
    autograd::Tensor train_step(
        const autograd::Tensor& input,
        const autograd::Tensor& target,
        std::function<autograd::Tensor(const autograd::Tensor&, const autograd::Tensor&)> loss_fn);
    std::vector<autograd::Tensor*> parameters();
    OptimizerType selected_optimizer_type() const noexcept { return optimizer_type_; }
    std::variant<std::monostate, SGD, Adam, RMSProp, NAdam>& optimizer() noexcept { return optimizer_; }
    bool using_ucao_engine() const noexcept { return ucao::engine::select_runtime(model_family()).selected && ucao_engine_.enabled; }
    ucao::engine::EngineDescriptor engine_descriptor() const noexcept { return ucao_engine_; }
    ucao::engine::SelectionResult engine_selection() const noexcept { return ucao::engine::select_runtime(model_family()); }
    static constexpr ucao::engine::ModelFamily model_family() noexcept { return ucao::engine::ModelFamily::Mlp; }
private:
    static OptimizerType auto_select_optimizer(const std::vector<autograd::Tensor*>& params,
                                               double learning_rate) noexcept;
    std::unique_ptr<Sequential> network_;
    OptimizerType optimizer_type_{OptimizerType::Auto};
    std::variant<std::monostate, SGD, Adam, RMSProp, NAdam> optimizer_;
    ucao::engine::EngineDescriptor ucao_engine_{};
};
} // namespace mlp
