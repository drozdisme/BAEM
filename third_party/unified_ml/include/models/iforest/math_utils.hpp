#pragma once

#include <cstddef>

namespace iforest {

/// Euler–Mascheroni constant.
constexpr double EULER_GAMMA = 0.5772156649015328606;

/**
 * Approximate harmonic number H(i) = ln(i) + gamma.
 * Returns 0 for i <= 0.
 */
double harmonic(double i);

/**
 * Average path length of an unsuccessful binary search tree search with n keys.
 *
 *   c(1)   = 0
 *   c(2)   = 1
 *   c(n>2) = 2 * H(n-1) - 2*(n-1)/n
 *
 * Used to normalise path lengths in the anomaly score formula.
 */
double c(std::size_t n);

} // namespace iforest
