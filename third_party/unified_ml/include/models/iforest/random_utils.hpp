#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace iforest {

/**
 * Draw `sample_size` elements from `population` without replacement.
 * Implements a partial Fisher–Yates shuffle on a working copy.
 *
 * @throws std::invalid_argument if sample_size > population.size().
 */
std::vector<std::size_t> sample_without_replacement(
    const std::vector<std::size_t>& population,
    std::size_t                     sample_size,
    std::mt19937&                   rng
);

/**
 * Draw a uniform random double from the half-open interval [lo, hi).
 */
double random_uniform(double lo, double hi, std::mt19937& rng);

} // namespace iforest
