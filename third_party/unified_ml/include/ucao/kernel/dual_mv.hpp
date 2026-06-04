#pragma once

#include "ucao/kernel/blade_kernel.hpp"

#include <array>
#include <cmath>
#include <cstring>

namespace ucao::kernel {

namespace detail {

template <int N>
[[nodiscard]] constexpr int grade_of(int blade) noexcept {
    int g = 0;
    for (int i = 0; i < N; ++i) {
        g += (blade >> i) & 1;
    }
    return g;
}

template <int N>
[[nodiscard]] constexpr int reversion_sign(int blade) noexcept {
    const int g = grade_of<N>(blade);
    return ((g * (g - 1) / 2) & 1) ? -1 : 1;
}

template <int N, int P, int Q>
inline void gp_dispatch(float* out, const float* a, const float* b) noexcept {
    static constexpr int D = 1 << N;
    if constexpr (D <= 16) {
        gp_avx512<N, P, Q>(out, a, b);
    } else if constexpr (D == 32) {
        gp_avx512_d32<P, Q>(out, a, b);
    } else {
        gp_single<N, P, Q>(out, a, b);
    }
}

} // namespace detail

/**
 * @brief Dual-number multivector for forward-mode differentiation.
 * @pre N = P + Q and D = 2^N <= 32.
 * @post Stores A = A_r + ε A_d with ε² = 0.
 */
template <int N, int P, int Q>
struct DualMV {
    static_assert(P + Q == N, "ucao: algebra dimensions must sum to N");
    static_assert(N <= 5, "ucao: N > 5 requires 64-wide SIMD, not implemented");

    static constexpr int D = 1 << N;

    alignas(64) float real[D]{};
    alignas(64) float dual[D]{};

    DualMV() = default;

    /**
     * @brief Embed coordinates as a grade-1 multivector and seed one tangent basis direction.
     * @pre x points to N scalar coordinates and 0 <= mu < N.
     * @post real[1<<i] = x[i], dual[1<<mu] = 1.
     */
    [[nodiscard]] static DualMV embed_coord(const float* x, int mu) noexcept {
        DualMV out;
        for (int i = 0; i < N; ++i) {
            out.real[1 << i] = x[i];
        }
        if (mu >= 0 && mu < N) {
            out.dual[1 << mu] = 1.0f;
        }
        return out;
    }

    /**
     * @brief Wrap a real multivector as a dual multivector with zero tangent part.
     * @pre r points to D real coefficients.
     * @post real = r and dual = 0.
     */
    [[nodiscard]] static DualMV from_real(const float* r) noexcept {
        DualMV out;
        std::memcpy(out.real, r, sizeof(real));
        return out;
    }

    /**
     * @brief Reverse both the real and dual multivector parts.
     * @pre This multivector is valid in Cl(p,q).
     * @post rev(A_r + εA_d) = rev(A_r) + ε rev(A_d).
     */
    [[nodiscard]] DualMV reversed() const noexcept {
        DualMV out;
        for (int i = 0; i < D; ++i) {
            const float s = static_cast<float>(detail::reversion_sign<N>(i));
            out.real[i] = s * real[i];
            out.dual[i] = s * dual[i];
        }
        return out;
    }

    /**
     * @brief Apply a Clifford sandwich W x ~W while propagating the tangent part through x.
     * @pre W points to D real coefficients.
     * @post Returns gp_d(gp_d(W, x), rev(W)).
     */
    [[nodiscard]] static DualMV sandwich(const float* W, const DualMV& x) noexcept {
        alignas(64) float rev_w[D]{};
        for (int i = 0; i < D; ++i) {
            rev_w[i] = static_cast<float>(detail::reversion_sign<N>(i)) * W[i];
        }
        const DualMV tmp = gp_d(from_real(W), x);
        return gp_d(tmp, from_real(rev_w));
    }

    /**
     * @brief Clifford norm with exact first derivative.
     * @pre The real part is finite.
     * @post real_out = real / sqrt(|n2| + eps), dual_out follows the chain rule.
     */
    [[nodiscard]] DualMV clifford_norm() const noexcept {
        constexpr float eps = 1e-8f;
        float n2 = 0.0f;
        float proj = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float rev = static_cast<float>(detail::reversion_sign<N>(i));
            n2 += rev * real[i] * real[i];
            proj += real[i] * dual[i];
        }
        const float inv = 1.0f / std::sqrt(std::fabs(n2) + eps);
        const float denom = n2 + eps;
        DualMV out;
        for (int i = 0; i < D; ++i) {
            out.real[i] = real[i] * inv;
            out.dual[i] = (dual[i] - real[i] * proj / denom) * inv;
        }
        return out;
    }

    /**
     * @brief Dual geometric product implementing the Leibniz rule.
     * @pre Both operands have D coefficients in real and dual parts.
     * @post C_r = A_r B_r and C_d = A_d B_r + A_r B_d.
     */
    [[nodiscard]] friend DualMV gp_d(const DualMV& A, const DualMV& B) noexcept {
        DualMV out;
        alignas(64) float tmp1[D]{};
        alignas(64) float tmp2[D]{};
        detail::gp_dispatch<N, P, Q>(out.real, A.real, B.real);
        detail::gp_dispatch<N, P, Q>(tmp1, A.dual, B.real);
        detail::gp_dispatch<N, P, Q>(tmp2, A.real, B.dual);
        for (int i = 0; i < D; ++i) {
            out.dual[i] = tmp1[i] + tmp2[i];
        }
        return out;
    }
};

} // namespace ucao::kernel
