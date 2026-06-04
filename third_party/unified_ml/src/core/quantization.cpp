#include "core/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace core {

namespace {

std::int8_t clamp_i8(int v) {
    if (v < -128) return static_cast<std::int8_t>(-128);
    if (v > 127) return static_cast<std::int8_t>(127);
    return static_cast<std::int8_t>(v);
}

}  // namespace

QuantParams calibrate_quant_params(const autograd::Tensor& x) {
    double min_v = x.value_flat(0), max_v = x.value_flat(0);
    for (std::size_t i = 1; i < x.numel(); ++i) {
        min_v = std::min(min_v, x.value_flat(i));
        max_v = std::max(max_v, x.value_flat(i));
    }
    if (min_v == max_v) return {1.0, 0};
    const double qmin = -128.0, qmax = 127.0;
    const double scale = (max_v - min_v) / (qmax - qmin);
    int zero_point = static_cast<int>(std::llround(qmin - min_v / scale));
    if (zero_point < -128) zero_point = -128;
    if (zero_point > 127) zero_point = 127;
    return {scale, zero_point};
}

QuantizedTensor quantize_per_tensor(const autograd::Tensor& x, QuantParams params) {
    QuantizedTensor q;
    q.shape = x.shape();
    q.params = params;
    q.values.resize(x.numel());
    for (std::size_t i = 0; i < x.numel(); ++i) {
        const int qv = static_cast<int>(std::llround(x.value_flat(i) / params.scale)) + params.zero_point;
        q.values[i] = clamp_i8(qv);
    }
    return q;
}

autograd::Tensor dequantize_per_tensor(const QuantizedTensor& q) {
    std::vector<double> out(q.values.size());
    for (std::size_t i = 0; i < q.values.size(); ++i)
        out[i] = (static_cast<int>(q.values[i]) - q.params.zero_point) * q.params.scale;
    return autograd::Tensor(std::move(out), q.shape, false);
}

QuantizedPerChannelTensor quantize_per_channel_symmetric(const autograd::Tensor& x) {
    if (x.ndim() != 2) throw std::invalid_argument("quantize_per_channel_symmetric: expected 2-D weight tensor");
    const std::size_t out_dim = x.shape()[0];
    const std::size_t in_dim = x.shape()[1];
    QuantizedPerChannelTensor q;
    q.shape = x.shape();
    q.zero_point = 0;
    q.scales.resize(out_dim, 1.0);
    q.values.resize(x.numel());
    for (std::size_t o = 0; o < out_dim; ++o) {
        double amax = 0.0;
        for (std::size_t i = 0; i < in_dim; ++i)
            amax = std::max(amax, std::abs(x.value_flat(o * in_dim + i)));
        const double scale = (amax == 0.0) ? 1.0 : (amax / 127.0);
        q.scales[o] = scale;
        for (std::size_t i = 0; i < in_dim; ++i) {
            const int qv = static_cast<int>(std::llround(x.value_flat(o * in_dim + i) / scale));
            q.values[o * in_dim + i] = clamp_i8(qv);
        }
    }
    return q;
}

autograd::Tensor quantized_linear(const autograd::Tensor& x,
                                  const QuantizedPerChannelTensor& weight,
                                  const autograd::Tensor& bias) {
    if (x.ndim() != 2) throw std::invalid_argument("quantized_linear: x must be [batch, in_dim]");
    if (weight.shape.size() != 2) throw std::invalid_argument("quantized_linear: weight must be 2-D");
    const std::size_t batch = x.shape()[0];
    const std::size_t in_dim = x.shape()[1];
    const std::size_t out_dim = weight.shape[0];
    if (weight.shape[1] != in_dim) throw std::invalid_argument("quantized_linear: input dim mismatch");
    if (bias.numel() != out_dim) throw std::invalid_argument("quantized_linear: bias size mismatch");

    auto x_params = calibrate_quant_params(x);
    auto xq = quantize_per_tensor(x, x_params);

    std::vector<double> out(batch * out_dim, 0.0);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < out_dim; ++o) {
            int acc = 0;
            for (std::size_t i = 0; i < in_dim; ++i) {
                const int xv = static_cast<int>(xq.values[b * in_dim + i]) - xq.params.zero_point;
                const int wv = static_cast<int>(weight.values[o * in_dim + i]);
                acc += xv * wv;
            }
            out[b * out_dim + o] = acc * xq.params.scale * weight.scales[o] + bias.value_flat(o);
        }
    }
    return autograd::Tensor(std::move(out), {batch, out_dim}, false);
}

}  // namespace core
