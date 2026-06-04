#pragma once
#include "autograd/autograd.h"
#include <cstddef>
#include <functional>
#include <random>
#include <vector>
namespace deep_onet {
struct Sample {
    std::vector<double> u_vals;
    std::vector<double> y_coord;
    double              target;
};
struct Dataset {
    std::vector<Sample> samples;
    std::size_t size()       const { return samples.size(); }
    std::size_t sensor_dim() const { return samples.empty() ? 0 : samples[0].u_vals.size(); }
    std::size_t coord_dim()  const { return samples.empty() ? 0 : samples[0].y_coord.size(); }
};
void make_batch(const Dataset& dataset, const std::vector<std::size_t>& indices,
                autograd::Tensor& u_out, autograd::Tensor& y_out, autograd::Tensor& t_out);
Dataset make_antiderivative_dataset(std::size_t n_samples, std::size_t n_sensors,
                                    std::size_t n_harmonics = 3, unsigned seed = 42);
Dataset make_identity_dataset(std::size_t n_samples, std::size_t n_sensors, unsigned seed = 42);
double mse_raw(const std::vector<double>& pred, const std::vector<double>& target);
double rel_l2_raw(const std::vector<double>& pred, const std::vector<double>& target);
std::vector<std::size_t> shuffled_indices(std::size_t n, std::mt19937& rng);
} // namespace deep_onet
