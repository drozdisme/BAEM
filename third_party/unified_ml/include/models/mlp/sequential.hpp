#pragma once
#include "models/mlp/layer.hpp"
#include <memory>
#include <vector>
namespace mlp {
class Sequential final : public Layer {
public:
    void add(std::unique_ptr<Layer> layer);
    autograd::Tensor forward(const autograd::Tensor& input) override;
    std::vector<autograd::Tensor*> parameters() override;
    std::size_t size()  const noexcept { return layers_.size(); }
    bool        empty() const noexcept { return layers_.empty(); }
private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
} // namespace mlp
