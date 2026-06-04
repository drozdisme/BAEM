#pragma once

#include "ucao/kernel/sign_table.hpp"

#include <array>
#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace ucao::kernel {

namespace detail {

template <int Mask>
[[nodiscard]] constexpr int grade() noexcept {
    int g = 0;
    for (int i = 0; i < 31; ++i) {
        g += (Mask >> i) & 1;
    }
    return g;
}

template <int N, int P, int Q>
[[nodiscard]] constexpr int rev_sign(int blade) noexcept {
    int g = 0;
    for (int i = 0; i < N; ++i) {
        g += (blade >> i) & 1;
    }
    return ((g * (g - 1) / 2) & 1) ? -1 : 1;
}

#ifdef __AVX512F__
template <int D>
struct ShufflePlan {
    std::array<int, 16> idx{};
    std::array<float, 16> sign{};
};

template <int N, int P, int Q>
[[nodiscard]] consteval auto make_shuffle_plans() {
    constexpr int D = 1 << N;
    std::array<ShufflePlan<D>, D> plans{};
    for (int c = 0; c < D; ++c) {
        for (int lane = 0; lane < 16; ++lane) {
            if (lane < D) {
                plans[c].idx[lane] = lane ^ c;
                plans[c].sign[lane] = static_cast<float>(gp_sign_cx<N, P, Q>(lane, lane ^ c));
            } else {
                plans[c].idx[lane] = 0;
                plans[c].sign[lane] = 0.0f;
            }
        }
    }
    return plans;
}

template <int N, int P, int Q>
inline constexpr auto kShufflePlans = make_shuffle_plans<N, P, Q>();

template <int N, int P, int Q>
[[gnu::target("avx512f,avx512vl"), gnu::flatten, gnu::noinline, gnu::optimize("unroll-loops")]]
inline void gp_single_avx512_shuffled(float* __restrict__ out,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b) noexcept {
    static constexpr int D = 1 << N;
    alignas(64) float a_pad[16]{};
    alignas(64) float b_pad[16]{};
    alignas(64) float lanes[16 * D]{};
    for (int i = 0; i < D; ++i) {
        a_pad[i] = a[i];
        b_pad[i] = b[i];
    }
    const __m512 va = _mm512_load_ps(a_pad);
    const __m512 vb = _mm512_load_ps(b_pad);
    for (int c = 0; c < D; ++c) {
        const __m512 rhs = _mm512_permutex2var_ps(va, _mm512_load_epi32(kShufflePlans<N, P, Q>[c].idx.data()), vb);
        const __m512 sgn = _mm512_load_ps(kShufflePlans<N, P, Q>[c].sign.data());
        const __m512 prod = _mm512_mul_ps(_mm512_mul_ps(va, rhs), sgn);
        _mm512_store_ps(lanes + c * 16, prod);
    }
    for (int c = 0; c < D; ++c) {
        float acc = 0.0f;
        for (int i = 0; i < D; ++i) {
            acc += lanes[c * 16 + i];
        }
        out[c] = acc;
    }
}
#endif

} // namespace detail

/**
 * @brief Scalar geometric product for a single multivector pair.
 * @pre out, a and b point to D = 2^N contiguous float values.
 * @post out[c] = sum_a sign(a, a^c) * a[a] * b[a^c].
 */
template <int N, int P, int Q>
[[gnu::flatten, gnu::noinline]] inline void gp_single(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b) noexcept {
    static constexpr int D = 1 << N;
#ifdef __AVX512F__
    if constexpr ((N == 3 && P == 3 && Q == 0) || (N == 4 && P == 4 && Q == 0) || (N == 4 && P == 3 && Q == 1)) {
        detail::gp_single_avx512_shuffled<N, P, Q>(out, a, b);
        return;
    }
#endif
    for (int c = 0; c < D; ++c) {
        float acc = 0.0f;
        const auto& entries = kGPTable<N, P, Q>.sparse_[c];
        const int nnz = kGPTable<N, P, Q>.nnz_[c];
        for (int i = 0; i < nnz; ++i) {
            const auto e = entries[i];
            acc += static_cast<float>(e.sign) * a[e.a] * b[e.a ^ c];
        }
        out[c] = acc;
    }
}

/**
 * @brief AVX-512 batched geometric product for D lanes.
 * @pre A, B, C store structure-of-arrays layout with 16 batch items per blade row.
 * @post C[row*16 + lane] contains the GP result for the lane.
 */
template <int N, int P, int Q>
[[gnu::flatten, gnu::noinline]] inline void gp_avx512(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B) noexcept {
#ifdef __AVX512F__
    static constexpr int D = 1 << N;
    for (int c = 0; c < D; ++c) {
        __m512 acc = _mm512_setzero_ps();
        const auto& entries = kGPTable<N, P, Q>.sparse_[c];
        const int nnz = kGPTable<N, P, Q>.nnz_[c];
        for (int i = 0; i < nnz; ++i) {
            const auto e = entries[i];
            const __m512 lhs = _mm512_loadu_ps(A + e.a * 16);
            const __m512 rhs = _mm512_loadu_ps(B + (e.a ^ c) * 16);
            const __m512 sgn = _mm512_set1_ps(static_cast<float>(e.sign));
            acc = _mm512_fmadd_ps(_mm512_mul_ps(lhs, rhs), sgn, acc);
        }
        _mm512_storeu_ps(C + c * 16, acc);
    }
#else
    gp_single<N, P, Q>(C, A, B);
#endif
}

template <int P, int Q>
[[gnu::flatten, gnu::noinline]] inline void gp_avx512_d32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B) noexcept {
#ifdef __AVX512F__
    alignas(64) float tmp_lo[16 * 16]{};
    alignas(64) float tmp_hi[16 * 16]{};
    gp_avx512<4, P, Q>(tmp_lo, A, B);
    gp_avx512<4, P, Q>(tmp_hi, A + 16 * 16, B + 16 * 16);
    std::memcpy(C, tmp_lo, sizeof(tmp_lo));
    std::memcpy(C + 16 * 16, tmp_hi, sizeof(tmp_hi));
#else
    gp_single<5, P, Q>(C, A, B);
#endif
}

} // namespace ucao::kernel
