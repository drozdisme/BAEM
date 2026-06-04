#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace core {

inline std::vector<std::uint64_t> threshold_pack(std::span<const double> values, double threshold) {
    const std::size_t word_count = (values.size() + 63U) / 64U;
    std::vector<std::uint64_t> packed(word_count, 0);
#if defined(__AVX512F__)
    constexpr std::size_t lanes = 8;
    alignas(64) double lane_values[lanes];
    std::size_t base = 0;
    while (base + lanes <= values.size()) {
        for (std::size_t lane = 0; lane < lanes; ++lane) lane_values[lane] = values[base + lane];
        const __m512d vals = _mm512_load_pd(lane_values);
        const __m512d thr = _mm512_set1_pd(threshold);
        const __mmask8 mask = _mm512_cmp_pd_mask(vals, thr, _CMP_GE_OQ);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if ((mask >> lane) & 0x1U) {
                const std::size_t idx = base + lane;
                packed[idx / 64U] |= (std::uint64_t{1} << (idx % 64U));
            }
        }
        base += lanes;
    }
    for (std::size_t i = base; i < values.size(); ++i) if (values[i] >= threshold) packed[i / 64U] |= (std::uint64_t{1} << (i % 64U));
#elif defined(__AVX2__)
    constexpr std::size_t lanes = 4;
    alignas(32) double lane_values[lanes];
    std::size_t base = 0;
    while (base + lanes <= values.size()) {
        for (std::size_t lane = 0; lane < lanes; ++lane) lane_values[lane] = values[base + lane];
        const __m256d vals = _mm256_load_pd(lane_values);
        const __m256d thr = _mm256_set1_pd(threshold);
        const __m256d cmp = _mm256_cmp_pd(vals, thr, _CMP_GE_OQ);
        const int mask = _mm256_movemask_pd(cmp);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if ((mask >> lane) & 0x1U) {
                const std::size_t idx = base + lane;
                packed[idx / 64U] |= (std::uint64_t{1} << (idx % 64U));
            }
        }
        base += lanes;
    }
    for (std::size_t i = base; i < values.size(); ++i) if (values[i] >= threshold) packed[i / 64U] |= (std::uint64_t{1} << (i % 64U));
#else
    for (std::size_t i = 0; i < values.size(); ++i) if (values[i] >= threshold) packed[i / 64U] |= (std::uint64_t{1} << (i % 64U));
#endif
    return packed;
}

inline std::vector<std::uint64_t> batch_to_bit_bridge(const double* values, std::size_t count, double threshold) {
    return threshold_pack(std::span<const double>(values, count), threshold);
}

} // namespace core
