#include "models/deep_onet/trunk_net.hpp"
#include <stdexcept>
namespace deep_onet {
TrunkNet::TrunkNet(std::size_t in, const std::vector<std::size_t>& hidden, std::size_t out, Activation act)
  : input_dim_(in), output_dim_(out), activation_(act)
{
  if (in == 0) throw std::invalid_argument("TrunkNet: input_dim must be > 0");
  if (out == 0) throw std::invalid_argument("TrunkNet: output_dim must be > 0");
  std::size_t prev = in;
  unsigned layer_index = 0;
  for (std::size_t h : hidden) { layers_.emplace_back(prev, h, true, layer_index++); prev = h; }
  layers_.emplace_back(prev, out, true, layer_index);
}
autograd::Tensor TrunkNet::forward(const autograd::Tensor& y) const {
  autograd::Tensor h = y;
  for (std::size_t i = 0; i + 1 < layers_.size(); ++i) {
    h = layers_[i].forward(h);
    h = apply_activation(h, activation_);
  }
  return layers_.back().forward(h);
}
std::vector<autograd::Tensor*> TrunkNet::parameters() {
  std::vector<autograd::Tensor*> all;
  for (auto& layer : layers_) { auto p = layer.parameters(); all.insert(all.end(), p.begin(), p.end()); }
  return all;
}
void TrunkNet::zero_grad() { for (auto& layer : layers_) layer.zero_grad(); }
} // namespace deep_onet
