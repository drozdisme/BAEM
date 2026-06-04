#include "models/iforest/math_utils.hpp"

#include <cmath>

namespace iforest {

double harmonic(double i) {
  if (i <= 0.0) return 0.0;
  return std::log(i) + EULER_GAMMA;
}

double c(std::size_t n) {
  if (n <= 1) return 0.0;
  if (n == 2) return 1.0;
  // c(n) = 2·H(n-1) − 2·(n-1)/n
  const double hn1 = harmonic(static_cast<double>(n - 1));
  return 2.0 * hn1
   - (2.0 * static_cast<double>(n - 1) / static_cast<double>(n));
}

} // namespace iforest
