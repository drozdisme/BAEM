#include "models/mlp/model.hpp"
#include <cstddef>
#include <stdexcept>
#include <type_traits>
namespace mlp {

OptimizerType Model::auto_select_optimizer(const std::vector<autograd::Tensor*>& params,
                                           double learning_rate) noexcept
{
    std::size_t total_params = 0;
    for (const auto* p : params) {
        if (p) total_params += p->numel();
    }

    // Conservative auto policy:
    //  - High learning rates remain stable with SGD.
    //  - Small networks benefit from NAdam's fast convergence.
    //  - Medium/large dense MLPs default to Adam.
    if (learning_rate >= 5e-2) return OptimizerType::SGD;
    if (total_params <= 2'000) return OptimizerType::NAdam;
    return OptimizerType::Adam;
}

Model::Model(std::unique_ptr<Sequential> network, double learning_rate)
    : Model(std::move(network), learning_rate, OptimizerType::Auto)
{}

Model::Model(std::unique_ptr<Sequential> network, double learning_rate, OptimizerType optimizer_type)
    : network_(std::move(network))
{
    if (!network_) throw std::invalid_argument("Model: null network");
    const auto selection = ucao::engine::select_runtime(ucao::engine::ModelFamily::Mlp);
    if (selection.selected) {
        ucao_engine_ = selection.descriptor;
    }
    auto params = network_->parameters();
    optimizer_type_ = (optimizer_type == OptimizerType::Auto)
                        ? auto_select_optimizer(params, learning_rate)
                        : optimizer_type;
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            optimizer_.emplace<1>(params, learning_rate);
            break;
        case OptimizerType::Adam:
            optimizer_.emplace<2>(params, learning_rate);
            break;
        case OptimizerType::RMSProp:
            optimizer_.emplace<3>(params, learning_rate);
            break;
        case OptimizerType::NAdam:
            optimizer_.emplace<4>(params, learning_rate);
            break;
        case OptimizerType::Auto:
            // Resolved into concrete optimizer above.
            break;
    }
}
autograd::Tensor Model::forward(const autograd::Tensor& input) { return network_->forward(input); }
autograd::Tensor Model::train_step(
    const autograd::Tensor& input,
    const autograd::Tensor& target,
    std::function<autograd::Tensor(const autograd::Tensor&, const autograd::Tensor&)> loss_fn)
{
    std::visit([](auto& opt) {
        using T = std::decay_t<decltype(opt)>;
        if constexpr (!std::is_same_v<T, std::monostate>) opt.zero_grad();
    }, optimizer_);
    autograd::Tensor output = network_->forward(input);
    autograd::Tensor loss   = loss_fn(output, target);
    loss.backward();
    std::visit([](auto& opt) {
        using T = std::decay_t<decltype(opt)>;
        if constexpr (!std::is_same_v<T, std::monostate>) opt.step();
    }, optimizer_);
    return loss;
}
std::vector<autograd::Tensor*> Model::parameters() { return network_->parameters(); }
} // namespace mlp
