#pragma once

#include "autograd/tensor.h"

#include <cstdint>
#include <vector>

namespace core {

struct QuantParams {
    double scale = 1.0;
    int zero_point = 0;
};

struct QuantizedTensor {
    std::vector<std::int8_t> values;
    std::vector<std::size_t> shape;
    QuantParams params;
};

struct QuantizedPerChannelTensor {
    std::vector<std::int8_t> values;
    std::vector<std::size_t> shape;
    std::vector<double> scales;
    int zero_point = 0;
};

QuantParams calibrate_quant_params(const autograd::Tensor& x);
QuantizedTensor quantize_per_tensor(const autograd::Tensor& x, QuantParams params);
autograd::Tensor dequantize_per_tensor(const QuantizedTensor& q);

QuantizedPerChannelTensor quantize_per_channel_symmetric(const autograd::Tensor& x);
autograd::Tensor quantized_linear(const autograd::Tensor& x,
                                  const QuantizedPerChannelTensor& weight,
                                  const autograd::Tensor& bias);

}  // namespace core
