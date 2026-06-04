#pragma once
#include "autograd/tensor.h"
#include <vector>

namespace mlp {
class Layer {
public:
    virtual ~Layer() = default;
    virtual autograd::Tensor forward(const autograd::Tensor& input) = 0;
    virtual std::vector<autograd::Tensor*> parameters() = 0;
    void zero_grad();
};
inline void Layer::zero_grad() {
    for (autograd::Tensor* p : parameters())
        if (p && p->requires_grad()) p->zero_grad();
}
} // namespace mlp
