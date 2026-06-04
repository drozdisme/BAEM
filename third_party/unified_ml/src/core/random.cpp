#include "core/random.hpp"

#include <stdexcept>

namespace core {

std::vector<std::size_t> sample_without_replacement(std::size_t n, std::size_t k, RNG& rng) {
  if (k > n) throw std::invalid_argument("sample_without_replacement: k > n");
  std::vector<std::size_t> pool = iota_indices(n);
  for (std::size_t i = 0; i < k; ++i) {
    std::size_t j = i + static_cast<std::size_t>(rng.uniform_int(0, static_cast<int>(n - i - 1)));
    std::swap(pool[i], pool[j]);
  }
  pool.resize(k);
  return pool;
}

std::vector<std::size_t> bootstrap_indices(std::size_t n, RNG& rng) {
  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i)
    idx[i] = static_cast<std::size_t>(rng.uniform_int(0, static_cast<int>(n) - 1));
  return idx;
}

std::vector<std::size_t> sample_population(const std::vector<std::size_t>& population,
               std::size_t k,
               std::mt19937& rng) {
  if (k > population.size())
    throw std::invalid_argument("sample_population: k > population size");
  std::vector<std::size_t> pool(population);
  for (std::size_t i = 0; i < k; ++i) {
    std::uniform_int_distribution<std::size_t> dist(i, pool.size() - 1);
    std::swap(pool[i], pool[dist(rng)]);
  }
  return std::vector<std::size_t>(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(k));
}

double random_uniform(double lo, double hi, std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(rng);
}

} // namespace core
