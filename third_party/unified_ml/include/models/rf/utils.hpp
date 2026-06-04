#include "core/random.hpp"
#pragma once

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rf {

// ---------------------------------------------------------------------------
// Random number generator wrapper (seeded, thread-local friendly)
// ---------------------------------------------------------------------------
class RNG {
public:
    explicit RNG(uint64_t seed = 42) : engine_(seed) {}

    void seed(uint64_t s) { engine_.seed(s); }

    // Uniform integer in [lo, hi]
    int uniform_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(engine_);
    }

    // Uniform real in [0, 1)
    double uniform_real() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine_);
    }

    // Shuffle a vector in-place
    template<typename T>
    void shuffle(std::vector<T>& v) {
        std::shuffle(v.begin(), v.end(), engine_);
    }

    std::mt19937_64& engine() { return engine_; }

private:
    std::mt19937_64 engine_;
};

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

// Build index vector [0, 1, ..., n-1]
inline std::vector<size_t> iota_indices(size_t n) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t(0));
    return idx;
}

// Sample k unique indices from [0, n) without replacement
inline std::vector<size_t> sample_without_replacement(size_t n, size_t k, RNG& rng) {
    if (k > n) throw std::invalid_argument("k > n in sample_without_replacement");
    std::vector<size_t> pool = iota_indices(n);
    for (size_t i = 0; i < k; ++i) {
        size_t j = i + rng.uniform_int(0, static_cast<int>(n - i - 1));
        std::swap(pool[i], pool[j]);
    }
    pool.resize(k);
    return pool;
}

// Bootstrap sample: n draws WITH replacement from [0, n)
inline std::vector<size_t> bootstrap_indices(size_t n, RNG& rng) {
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) {
        idx[i] = static_cast<size_t>(rng.uniform_int(0, static_cast<int>(n) - 1));
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

inline double safe_log2(double x) {
    return (x <= 0.0) ? 0.0 : std::log2(x);
}

inline double mean(const std::vector<double>& v, const std::vector<size_t>& idx) {
    if (idx.empty()) return 0.0;
    double s = 0.0;
    for (size_t i : idx) s += v[i];
    return s / static_cast<double>(idx.size());
}

inline double variance(const std::vector<double>& v, const std::vector<size_t>& idx) {
    if (idx.size() < 2) return 0.0;
    double m = mean(v, idx);
    double s = 0.0;
    for (size_t i : idx) {
        double d = v[i] - m;
        s += d * d;
    }
    return s / static_cast<double>(idx.size());
}

// Count occurrences of each integer class in [0, n_classes)
inline std::vector<int> class_counts(const std::vector<double>& y,
                                     const std::vector<size_t>& idx,
                                     int n_classes) {
    std::vector<int> counts(n_classes, 0);
    for (size_t i : idx) {
        int c = static_cast<int>(y[i]);
        if (c >= 0 && c < n_classes) ++counts[c];
    }
    return counts;
}

// Most frequent class label
inline int majority_class(const std::vector<double>& y, const std::vector<size_t>& idx) {
    if (idx.empty()) return 0;
    // Find range
    int max_class = 0;
    for (size_t i : idx) max_class = std::max(max_class, static_cast<int>(y[i]));
    std::vector<int> counts(max_class + 1, 0);
    for (size_t i : idx) ++counts[static_cast<int>(y[i])];
    return static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
}

} // namespace rf
