#include "models/mlp/sequential.hpp"
#include <stdexcept>
namespace mlp {
void Sequential::add(std::unique_ptr<Layer> layer) {
  if (!layer) throw std::invalid_argument("Sequential::add: null layer");
  layers_.push_back(std::move(layer));
}
autograd::Tensor Sequential::forward(const autograd::Tensor& input) {
  if (layers_.empty()) throw std::runtime_error("Sequential::forward: no layers");
  autograd::Tensor out = layers_.front()->forward(input);
  for (std::size_t i = 1; i < layers_.size(); ++i) out = layers_[i]->forward(out);
  return out;
}
std::vector<autograd::Tensor*> Sequential::parameters() {
  std::vector<autograd::Tensor*> all;
  for (auto& layer : layers_)
    for (autograd::Tensor* p : layer->parameters()) all.push_back(p);
  return all;
}
} // namespace mlp
