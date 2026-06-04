#include "models/iforest/random_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace iforest {

std::vector<std::size_t> sample_without_replacement(
  const std::vector<std::size_t>& population,
  std::size_t       sample_size,
  std::mt19937&       rng)
{
  if (sample_size > population.size()) {
    throw std::invalid_argument(
    "sample_size cannot exceed the population size.");
  }

  // Partial Fisher–Yates: swap first `sample_size` elements into place.
  std::vector<std::size_t> pool(population);

  for (std::size_t i = 0; i < sample_size; ++i) {
    std::uniform_int_distribution<std::size_t> dist(i, pool.size() - 1);
    std::swap(pool[i], pool[dist(rng)]);
  }

  return std::vector<std::size_t>(
    pool.begin(),
    pool.begin() + static_cast<std::ptrdiff_t>(sample_size)
  );
}

double random_uniform(double lo, double hi, std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(rng);
}

} // namespace iforest
