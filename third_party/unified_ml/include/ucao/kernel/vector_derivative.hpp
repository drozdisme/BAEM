#pragma once

#include "ucao/kernel/dual_mv.hpp"

#include <array>

namespace ucao::kernel {

/**
 * @brief Exact forward-mode vector derivative in Clifford coordinates.
 * @pre Layer exposes forward_dual(const DualMV<N,P,Q>&) -> DualMV<N,P,Q>.
 * @post Returns ∇F = sum_mu e_mu * (∂F/∂x_mu) using exactly N dual forward passes.
 */
template <int N, int P, int Q>
struct VectorDerivative {
    static constexpr int D = 1 << N;

    template <typename Layer>
    [[nodiscard]] static std::array<float, D> apply(const Layer& layer, const float* x) noexcept {
        std::array<float, D> result{};
        for (int mu = 0; mu < N; ++mu) {
            const DualMV<N, P, Q> input = DualMV<N, P, Q>::embed_coord(x, mu);
            const DualMV<N, P, Q> out = layer.forward_dual(input);
            result[1 << mu] = out.dual[0];
        }
        return result;
    }
};

} // namespace ucao::kernel
