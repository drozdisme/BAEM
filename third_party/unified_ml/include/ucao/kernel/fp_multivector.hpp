#pragma once

#include "ucao/kernel/sign_table.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ucao::kernel {

static_assert(sizeof(__int128) == 16, "ucao: __int128 must be 128-bit");

using fp64 = int64_t;
inline constexpr int FP_LOG2 = 30;
inline constexpr fp64 FP_SCALE = fp64(1) << FP_LOG2;

[[nodiscard]] inline fp64 fp_mul(fp64 a, fp64 b) noexcept {
    return static_cast<fp64>((static_cast<__int128>(a) * static_cast<__int128>(b)) >> FP_LOG2);
}

[[nodiscard]] inline fp64 fp_add(fp64 a, fp64 b) noexcept {
    return a + b;
}

[[nodiscard]] inline fp64 fp_inv_sqrt_nr(fp64 x_q30, int iters = 3) noexcept {
    const float x_float = static_cast<float>(x_q30) / static_cast<float>(FP_SCALE);
    fp64 y = static_cast<fp64>(static_cast<double>(FP_SCALE) / std::sqrt(static_cast<double>(x_float)));
    const __int128 S2 = static_cast<__int128>(FP_SCALE) * static_cast<__int128>(FP_SCALE);
    for (int i = 0; i < iters; ++i) {
        const __int128 y2_q60 = static_cast<__int128>(y) * static_cast<__int128>(y);
        const __int128 xy2_q60 = (static_cast<__int128>(x_q30) * y2_q60) >> FP_LOG2;
        const __int128 numer = static_cast<__int128>(y) * (3 * S2 - xy2_q60);
        const __int128 denom = 2 * S2;
        y = static_cast<fp64>(numer / denom);
    }
    return y;
}

namespace detail {

template <int N>
[[nodiscard]] constexpr int grade_of_fp(int blade) noexcept {
    int g = 0;
    for (int i = 0; i < N; ++i) {
        g += (blade >> i) & 1;
    }
    return g;
}

template <int N>
[[nodiscard]] constexpr int reversion_sign_fp(int blade) noexcept {
    const int g = grade_of_fp<N>(blade);
    return ((g * (g - 1) / 2) & 1) ? -1 : 1;
}

} // namespace detail

template <int N, int P, int Q, int R = 0>
struct FPMultivector {
    static_assert(P + Q + R == N, "ucao: algebra dimensions must sum to N");
    static_assert(N <= 5, "ucao: N > 5 requires 64-wide SIMD, not implemented");

    static constexpr int D = 1 << N;
    alignas(64) std::array<fp64, D> comps{};

    [[nodiscard]] static FPMultivector from_float(const float* data) noexcept {
        FPMultivector out;
        for (int i = 0; i < D; ++i) {
            out.comps[i] = static_cast<fp64>(std::llround(static_cast<double>(data[i]) * static_cast<double>(FP_SCALE)));
        }
        return out;
    }

    void to_float(float* out) const noexcept {
        for (int i = 0; i < D; ++i) {
            out[i] = static_cast<float>(static_cast<double>(comps[i]) / static_cast<double>(FP_SCALE));
        }
    }

    [[nodiscard]] FPMultivector reversed() const noexcept {
        FPMultivector out;
        for (int i = 0; i < D; ++i) {
            out.comps[i] = static_cast<fp64>(detail::reversion_sign_fp<N>(i)) * comps[i];
        }
        return out;
    }

    [[nodiscard]] __int128 norm_sq_q60() const noexcept {
        __int128 acc = 0;
        for (int i = 0; i < D; ++i) {
            acc += static_cast<__int128>(detail::reversion_sign_fp<N>(i)) *
                   static_cast<__int128>(comps[i]) * static_cast<__int128>(comps[i]);
        }
        return acc;
    }

    [[nodiscard]] bool is_unit_rotor(fp64 tol_q30 = fp64(1) << 10) const noexcept {
        const fp64 norm_q30 = static_cast<fp64>(norm_sq_q60() >> FP_LOG2);
        const fp64 diff = norm_q30 >= FP_SCALE ? (norm_q30 - FP_SCALE) : (FP_SCALE - norm_q30);
        return diff < tol_q30;
    }

    void reorthogonalize(int iters = 3) noexcept {
        if (is_unit_rotor()) {
            return;
        }
        const fp64 norm_q30 = static_cast<fp64>(norm_sq_q60() >> FP_LOG2);
        if (norm_q30 <= 0) {
            return;
        }
        const fp64 inv = fp_inv_sqrt_nr(norm_q30, iters);
        for (int i = 0; i < D; ++i) {
            comps[i] = static_cast<fp64>((static_cast<__int128>(comps[i]) * inv) >> FP_LOG2);
        }
    }

    void stabilize_rotor() noexcept {
        FPMultivector rev = reversed();
        FPMultivector rr = fp_gp(*this, rev);
        FPMultivector corr;
        corr.comps[0] = (3 * FP_SCALE) / 2 - rr.comps[0] / 2;
        for (int i = 1; i < D; ++i) {
            corr.comps[i] = -rr.comps[i] / 2;
        }
        *this = fp_gp(*this, corr);
        reorthogonalize(2);
    }

    [[nodiscard]] float rotor_error() const noexcept {
        const double norm = static_cast<double>(norm_sq_q60()) / static_cast<double>(FP_SCALE) / static_cast<double>(FP_SCALE);
        return static_cast<float>(std::fabs(norm - 1.0));
    }
};

template <int N, int P, int Q, int R>
[[nodiscard]] FPMultivector<N, P, Q, R> fp_gp(const FPMultivector<N, P, Q, R>& A,
                                              const FPMultivector<N, P, Q, R>& B) noexcept {
    FPMultivector<N, P, Q, R> C;
    static constexpr int D = 1 << N;
    for (int c = 0; c < D; ++c) {
        __int128 acc = 0;
        const auto& entries = kGPTable<N, P, Q, R>.sparse_[c];
        const int nnz = kGPTable<N, P, Q, R>.nnz_[c];
        for (int i = 0; i < nnz; ++i) {
            const auto e = entries[i];
            acc += static_cast<__int128>(e.sign) *
                   static_cast<__int128>(A.comps[e.a]) *
                   static_cast<__int128>(B.comps[e.a ^ c]);
        }
        C.comps[c] = static_cast<fp64>(acc >> FP_LOG2);
    }
    return C;
}

} // namespace ucao::kernel
