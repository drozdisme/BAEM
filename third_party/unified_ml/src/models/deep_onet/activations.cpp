#include "models/deep_onet/activations.hpp"
#include <algorithm>
#include <stdexcept>
namespace deep_onet {
Activation activation_from_string(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (lower == "relu")     return Activation::ReLU;
    if (lower == "tanh")     return Activation::Tanh;
    if (lower == "sigmoid")  return Activation::Sigmoid;
    if (lower == "none" || lower == "identity") return Activation::None;
    throw std::invalid_argument("Unknown activation: '" + name + "'");
}
} // namespace deep_onet
