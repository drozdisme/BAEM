#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>   // std::iota
#include <random>
#include <stdexcept>
#include <vector>

namespace core {

// ============================================================================
// RNG — seeded Mersenne Twister wrapper (64-bit engine).
// ============================================================================
class RNG {
public:
    explicit RNG(uint64_t seed = 42) : engine_(seed) {}
    void seed(uint64_t s) { engine_.seed(s); }

    /// Uniform integer in [lo, hi].
    int uniform_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(engine_);
    }

    /// Uniform real in [lo, hi).
    double uniform_real(double lo = 0.0, double hi = 1.0) {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(engine_);
    }

    template<typename T>
    void shuffle(std::vector<T>& v) {
        std::shuffle(v.begin(), v.end(), engine_);
    }

    std::mt19937_64& engine() { return engine_; }

private:
    std::mt19937_64 engine_;
};

// ============================================================================
// Sampling utilities
// ============================================================================

/// Build index vector [0, 1, ..., n-1].
inline std::vector<std::size_t> iota_indices(std::size_t n) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    return idx;
}

/// Sample k unique indices from [0, n) without replacement (partial Fisher–Yates).
std::vector<std::size_t> sample_without_replacement(std::size_t n, std::size_t k, RNG& rng);

/// Draw n samples WITH replacement from [0, n) — bootstrap.
std::vector<std::size_t> bootstrap_indices(std::size_t n, RNG& rng);

/// Draw k items from `population` without replacement.
std::vector<std::size_t> sample_population(const std::vector<std::size_t>& population,
                                            std::size_t k,
                                            std::mt19937& rng);

/// Draw uniform real from [lo, hi).
double random_uniform(double lo, double hi, std::mt19937& rng);

} // namespace core