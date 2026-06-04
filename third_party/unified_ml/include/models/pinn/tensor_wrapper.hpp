// tensor_wrapper.h — Convenience utilities built on top of autograd::Tensor.
//
// Provides:
//   linspace, zeros, ones, scalar_tensor, random initialization helpers,
//   and projection utilities for multi-dimensional PINN inputs.
//
// No modifications to autograd_engine are made; this is a pure wrapper layer.

#pragma once

#include "autograd/autograd.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace pinn {

//   Tensor creation utilities                         

/// Create a 1-D tensor of n evenly-spaced values in [lo, hi].
inline autograd::Tensor linspace(double lo, double hi, std::size_t n,
                                  bool requires_grad = false)
{
    if (n == 0) throw std::invalid_argument("linspace: n must be > 0");
    std::vector<double> data(n);
    if (n == 1) {
        data[0] = lo;
    } else {
        double step = (hi - lo) / static_cast<double>(n - 1);
        for (std::size_t i = 0; i < n; ++i)
            data[i] = lo + step * static_cast<double>(i);
    }
    return autograd::Tensor(data, {n}, requires_grad);
}

/// Create a tensor filled with zeros.
inline autograd::Tensor zeros(std::vector<std::size_t> shape,
                               bool requires_grad = false)
{
    std::size_t total = 1;
    for (auto d : shape) total *= d;
    return autograd::Tensor(std::vector<double>(total, 0.0), shape, requires_grad);
}

/// Create a tensor filled with ones.
inline autograd::Tensor ones(std::vector<std::size_t> shape,
                              bool requires_grad = false)
{
    std::size_t total = 1;
    for (auto d : shape) total *= d;
    return autograd::Tensor(std::vector<double>(total, 1.0), shape, requires_grad);
}

/// Create a scalar tensor from a double value.
inline autograd::Tensor scalar_tensor(double val, bool requires_grad = false)
{
    return autograd::Tensor(val, requires_grad);
}

/// Create a [1, n] row tensor from a std::vector.
inline autograd::Tensor row_tensor(std::vector<double> vals,
                                    bool requires_grad = false)
{
    std::size_t n = vals.size();
    return autograd::Tensor(std::move(vals), {1, n}, requires_grad);
}

/// Create a [n, m] tensor filled with uniform random values in [lo, hi].
inline autograd::Tensor rand_uniform(std::size_t n, std::size_t m,
                                      double lo, double hi,
                                      std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(lo, hi);
    std::vector<double> data(n * m);
    for (auto& v : data) v = dist(rng);
    return autograd::Tensor(data, {n, m}, /*requires_grad=*/false);
}

/// Create a [n, m] tensor filled with normal random values (mean, std).
inline autograd::Tensor rand_normal(std::size_t n, std::size_t m,
                                     double mean, double std_dev,
                                     std::mt19937& rng)
{
    std::normal_distribution<double> dist(mean, std_dev);
    std::vector<double> data(n * m);
    for (auto& v : data) v = dist(rng);
    return autograd::Tensor(data, {n, m}, /*requires_grad=*/false);
}

/// Xavier uniform initialization for a [fan_in, fan_out] weight matrix.
/// Returns a Tensor with requires_grad = true.
inline autograd::Tensor xavier_uniform(std::size_t fan_in, std::size_t fan_out,
                                        std::mt19937& rng)
{
    double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    std::uniform_real_distribution<double> dist(-limit, limit);
    std::vector<double> data(fan_in * fan_out);
    for (auto& v : data) v = dist(rng);
    return autograd::Tensor(data, {fan_in, fan_out}, /*requires_grad=*/true);
}

/// Zero-initialize a bias vector [1, size] with requires_grad = true.
inline autograd::Tensor zero_bias(std::size_t size)
{
    return autograd::Tensor(std::vector<double>(size, 0.0), {1, size},
                             /*requires_grad=*/true);
}

//   Multi-dimensional projection helpers                    
// These allow extracting a scalar partial derivative from a full gradient tensor
// using differentiable Tensor arithmetic (no custom ops required).
//
// Usage for input xt = [x, t] with shape {1, 2}:
//   Tensor e_x = project_mask(2, 0);   // [1, 2] = {1, 0}
//   Tensor e_t = project_mask(2, 1);   // [1, 2] = {0, 1}
//   Tensor du_dx = autograd::sum(autograd::mul(du_dxt, e_x));
//   Tensor du_dt = autograd::sum(autograd::mul(du_dxt, e_t));

/// Create a one-hot projection mask of shape {1, dim} selecting component idx.
inline autograd::Tensor project_mask(std::size_t dim, std::size_t idx)
{
    if (idx >= dim)
        throw std::out_of_range("project_mask: idx >= dim");
    std::vector<double> data(dim, 0.0);
    data[idx] = 1.0;
    return autograd::Tensor(data, {1, dim}, /*requires_grad=*/false);
}

//   Numerical helpers                             

/// Clamp a value to [lo, hi].
inline double clamp(double x, double lo, double hi)
{
    return std::max(lo, std::min(hi, x));
}

/// Compute element-wise absolute error statistics between two std::vectors.
struct ErrorStats {
    double max_abs;
    double mean_abs;
    double rel_max;
};

inline ErrorStats error_stats(const std::vector<double>& pred,
                               const std::vector<double>& ref)
{
    if (pred.size() != ref.size())
        throw std::invalid_argument("error_stats: size mismatch");
    double max_abs = 0.0, sum_abs = 0.0, max_rel = 0.0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        double ae = std::abs(pred[i] - ref[i]);
        double re = ae / std::max(1.0, std::abs(ref[i]));
        max_abs = std::max(max_abs, ae);
        sum_abs += ae;
        max_rel = std::max(max_rel, re);
    }
    return {max_abs, sum_abs / pred.size(), max_rel};
}

}  // namespace pinn
