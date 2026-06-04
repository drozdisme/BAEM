#pragma once

#include <array>
#include <cstdint>

namespace ucao::kernel {

/**
 * @brief Compute the Clifford geometric-product sign for two basis blades.
 * @pre P + Q + R == N and N <= 5.
 * @post Returns {-1, 0, +1} where 0 denotes a null-vector overlap.
 *
 * The sign is computed by permutation parity plus metric signature:
 *   sign(A,B) = (-1)^{swaps(A,B)} * metric(A & B)
 */
template <int N, int P, int Q, int R = 0>
[[nodiscard]] constexpr int gp_sign_cx(int A, int B) noexcept {
    static_assert(P + Q + R == N, "ucao: algebra dimensions must sum to N");
    static_assert(N <= 5, "ucao: N > 5 requires 64-wide SIMD, not implemented");

    int sign = 1;
    for (int i = N - 1; i >= 0; --i) {
        if (((A >> i) & 1) == 0) {
            continue;
        }
        int lower_count = 0;
        for (int j = 0; j < i; ++j) {
            lower_count += (B >> j) & 1;
        }
        if ((lower_count & 1) != 0) {
            sign = -sign;
        }
    }

    const int common = A & B;
    for (int i = 0; i < N; ++i) {
        if (((common >> i) & 1) == 0) {
            continue;
        }
        if (i < P) {
            continue;
        }
        if (i < P + Q) {
            sign = -sign;
            continue;
        }
        return 0;
    }

    return sign;
}

struct GPEntry {
    int16_t a;
    int8_t sign;
    int8_t _pad = 0;
};

/**
 * @brief Compile-time Clifford multiplication sign table and sparse routing table.
 * @pre P + Q + R == N and D <= 32.
 * @post sign_[a][b] stores the full GP sign, sparse_[c] stores contributing blades.
 */
template <int N, int P, int Q, int R = 0>
struct GPSignTable {
    static_assert(P + Q + R == N, "ucao: algebra dimensions must sum to N");
    static_assert(N <= 5, "ucao: N > 5 requires 64-wide SIMD, not implemented");

    static constexpr int D = 1 << N;
    static_assert(D <= 32, "ucao: D > 32 is not supported");

    std::array<std::array<int8_t, D>, D> sign_{};
    std::array<std::array<GPEntry, D>, D> sparse_{};
    std::array<int, D> nnz_{};

    constexpr GPSignTable() noexcept {
        for (int a = 0; a < D; ++a) {
            for (int b = 0; b < D; ++b) {
                const auto s = static_cast<int8_t>(gp_sign_cx<N, P, Q, R>(a, b));
                sign_[a][b] = s;
                if (s == 0) {
                    continue;
                }
                const int c = a ^ b;
                const int idx = nnz_[c]++;
                sparse_[c][idx] = GPEntry{static_cast<int16_t>(a), s, 0};
            }
        }
    }

    [[nodiscard]] constexpr int8_t sign(int a, int b) const noexcept {
        return sign_[a][b];
    }
};

template <int N, int P, int Q, int R = 0>
inline constexpr GPSignTable<N, P, Q, R> kGPTable{};

} // namespace ucao::kernel

static_assert(ucao::kernel::kGPTable<3, 3, 0>.sign(0b001, 0b010) == +1);
