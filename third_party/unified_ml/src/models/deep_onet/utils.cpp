#include "models/deep_onet/utils.hpp"
// utils.cpp — Dataset helpers and synthetic data generators.
//
// Antiderivative dataset
//    
// Input function: u(x) = Σ_{k=1}^{K} a_k * sin(k*π*x) + b_k * cos(k*π*x)
// Coefficients a_k, b_k drawn uniformly from [-1, 1].
// Sensor points: x_i = i / (m-1) for i=0,...,m-1  (uniform on [0,1]).
// Query point y: drawn uniformly from (0, 1].
// Target:    G(u)(y) = ∫_0^y u(x) dx
//        = Σ_k [ a_k/kπ * (-cos(kπy) + 1) + b_k/kπ * sin(kπy) ]
//
// Identity dataset
//   
// Target: G(u)(y) = u(y), where y is one of the sensor points.
// This tests whether the network can learn a trivial interpolation.

#include "models/deep_onet/utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace deep_onet {

// ═══════════════════════════════════════════════════════════════════════════════
//  make_batch
// ═══════════════════════════════════════════════════════════════════════════════

void make_batch(const Dataset&      dataset,
      const std::vector<std::size_t>& indices,
      autograd::Tensor&     u_out,
      autograd::Tensor&     y_out,
      autograd::Tensor&     t_out)
{
  if (indices.empty())
    throw std::invalid_argument("make_batch: indices must not be empty");

  const std::size_t batch = indices.size();
  const std::size_t m   = dataset.sensor_dim();
  const std::size_t d   = dataset.coord_dim();

  std::vector<double> u_data(batch * m);
  std::vector<double> y_data(batch * d);
  std::vector<double> t_data(batch * 1);

  for (std::size_t i = 0; i < batch; ++i) {
    const auto& s = dataset.samples[indices[i]];

    // u row
    for (std::size_t j = 0; j < m; ++j)
    u_data[i * m + j] = s.u_vals[j];

    // y row
    for (std::size_t j = 0; j < d; ++j)
    y_data[i * d + j] = s.y_coord[j];

    // target
    t_data[i] = s.target;
  }

  u_out = autograd::Tensor(std::move(u_data), {batch, m},   false);
  y_out = autograd::Tensor(std::move(y_data), {batch, d},   false);
  t_out = autograd::Tensor(std::move(t_data), {batch, 1},   false);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  make_antiderivative_dataset
// ═══════════════════════════════════════════════════════════════════════════════

Dataset make_antiderivative_dataset(std::size_t n_samples,
            std::size_t n_sensors,
            std::size_t n_harmonics,
            unsigned  seed)
{
  if (n_samples == 0) throw std::invalid_argument("n_samples must be > 0");
  if (n_sensors < 2)  throw std::invalid_argument("n_sensors must be >= 2");

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> coeff_dist(-1.0,  1.0);
  std::uniform_real_distribution<double> y_dist  ( 0.01, 0.99);

  // Sensor points uniformly spaced on [0, 1].
  std::vector<double> sensors(n_sensors);
  for (std::size_t i = 0; i < n_sensors; ++i)
    sensors[i] = static_cast<double>(i) / static_cast<double>(n_sensors - 1);

  Dataset ds;
  ds.samples.reserve(n_samples);

  for (std::size_t s = 0; s < n_samples; ++s) {
    // Sample random Fourier coefficients.
    std::vector<double> a(n_harmonics), b(n_harmonics);
    for (std::size_t k = 0; k < n_harmonics; ++k) {
    a[k] = coeff_dist(rng);
    b[k] = coeff_dist(rng);
    }

    // Evaluate u(x) at sensor points.
    std::vector<double> u_vals(n_sensors);
    for (std::size_t i = 0; i < n_sensors; ++i) {
    double x = sensors[i];
    double val = 0.0;
    for (std::size_t k = 0; k < n_harmonics; ++k) {
      double freq = static_cast<double>(k + 1) * M_PI;
      val += a[k] * std::sin(freq * x) + b[k] * std::cos(freq * x);
    }
    u_vals[i] = val;
    }

    // Query point y ~ Uniform(0.01, 0.99).
    double y = y_dist(rng);

    // Compute ∫_0^y u(x) dx analytically:
    // ∫_0^y a_k sin(kπx) dx = a_k/kπ * (1 - cos(kπy))
    // ∫_0^y b_k cos(kπx) dx = b_k/kπ * sin(kπy)
    double target = 0.0;
    for (std::size_t k = 0; k < n_harmonics; ++k) {
    double freq = static_cast<double>(k + 1) * M_PI;
    target += (a[k] / freq) * (1.0 - std::cos(freq * y));
    target += (b[k] / freq) * std::sin(freq * y);
    }

    Sample sample;
    sample.u_vals  = std::move(u_vals);
    sample.y_coord = { y };
    sample.target  = target;
    ds.samples.push_back(std::move(sample));
  }

  return ds;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  make_identity_dataset
// ═══════════════════════════════════════════════════════════════════════════════

Dataset make_identity_dataset(std::size_t n_samples,
          std::size_t n_sensors,
          unsigned  seed)
{
  if (n_samples == 0) throw std::invalid_argument("n_samples must be > 0");
  if (n_sensors < 2)  throw std::invalid_argument("n_sensors must be >= 2");

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double>  coeff_dist(-1.0, 1.0);
  std::uniform_int_distribution<std::size_t>  idx_dist(0, n_sensors - 1);

  std::vector<double> sensors(n_sensors);
  for (std::size_t i = 0; i < n_sensors; ++i)
    sensors[i] = static_cast<double>(i) / static_cast<double>(n_sensors - 1);

  Dataset ds;
  ds.samples.reserve(n_samples);

  for (std::size_t s = 0; s < n_samples; ++s) {
    // Random smooth function: sum of sinusoids.
    const std::size_t K = 3;
    std::vector<double> a(K), b(K);
    for (std::size_t k = 0; k < K; ++k) {
    a[k] = coeff_dist(rng);
    b[k] = coeff_dist(rng);
    }

    std::vector<double> u_vals(n_sensors);
    for (std::size_t i = 0; i < n_sensors; ++i) {
    double x = sensors[i], val = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
      double freq = static_cast<double>(k + 1) * M_PI;
      val += a[k] * std::sin(freq * x) + b[k] * std::cos(freq * x);
    }
    u_vals[i] = val;
    }

    // Query at one of the sensor locations.
    std::size_t qi = idx_dist(rng);
    double y   = sensors[qi];
    double target  = u_vals[qi];

    Sample sample;
    sample.u_vals  = std::move(u_vals);
    sample.y_coord = { y };
    sample.target  = target;
    ds.samples.push_back(std::move(sample));
  }

  return ds;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metrics
// ═══════════════════════════════════════════════════════════════════════════════

double mse_raw(const std::vector<double>& pred,
     const std::vector<double>& target)
{
  assert(pred.size() == target.size());
  double s = 0.0;
  for (std::size_t i = 0; i < pred.size(); ++i) {
    double d = pred[i] - target[i];
    s += d * d;
  }
  return s / static_cast<double>(pred.size());
}

double rel_l2_raw(const std::vector<double>& pred,
      const std::vector<double>& target)
{
  double num = 0.0, den = 1e-8;
  for (std::size_t i = 0; i < pred.size(); ++i) {
    double d = pred[i] - target[i];
    num += d * d;
    den += target[i] * target[i];
  }
  return std::sqrt(num / den);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Shuffled indices
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<std::size_t> shuffled_indices(std::size_t n, std::mt19937& rng)
{
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);
  return idx;
}

} // namespace deep_onet
