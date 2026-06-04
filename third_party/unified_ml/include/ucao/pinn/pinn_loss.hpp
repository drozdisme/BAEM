#pragma once

#include "autograd/autograd.h"
#include "ucao/kernel/blade_kernel.hpp"
#include "ucao/kernel/vector_derivative.hpp"
#include "ucao/pinn/field_layer.hpp"

namespace ucao::pinn {

/**
 * @brief Exact PINN residual loss using Clifford vector derivatives.
 * @pre collocation_pts has n_pts * N floats, source_J has n_pts * D floats, n_pts > 0.
 * @post Returns the mean scalar residual norm squared.
 */
template <int N, int P, int Q, int HiddenDim = 64>
[[nodiscard]] float clifford_pinn_loss_exact(const CliffordFieldLayer<N, P, Q, HiddenDim>& layer,
                                             const float* collocation_pts,
                                             const float* source_J,
                                             int n_pts) noexcept {
    static constexpr int D = 1 << N;
    if (n_pts <= 0) {
        return 0.0f;
    }
    float loss = 0.0f;
    for (int idx = 0; idx < n_pts; ++idx) {
        alignas(64) float R[D]{};
        alignas(64) float revR[D]{};
        alignas(64) float RrevR[D]{};
        const float* x = collocation_pts + idx * N;
        const float* J = source_J + idx * D;
        const auto nablaF = ucao::kernel::VectorDerivative<N, P, Q>::template apply<CliffordFieldLayer<N, P, Q, HiddenDim>>(layer, x);
        for (int i = 0; i < D; ++i) {
            R[i] = nablaF[i] - J[i];
            revR[i] = static_cast<float>(ucao::kernel::detail::reversion_sign<N>(i)) * R[i];
        }
        ucao::kernel::gp_single<N, P, Q>(RrevR, R, revR);
        loss += RrevR[0];
    }
    return loss / static_cast<float>(n_pts);
}

/**
 * @brief Wrap the exact PINN loss as a scalar autograd tensor.
 * @pre Same preconditions as clifford_pinn_loss_exact().
 * @post Returns a scalar tensor with requires_grad=false.
 */
template <int N, int P, int Q, int HiddenDim = 64>
[[nodiscard]] autograd::Tensor clifford_pinn_loss_tensor(const CliffordFieldLayer<N, P, Q, HiddenDim>& layer,
                                                         const float* collocation_pts,
                                                         const float* source_J,
                                                         int n_pts) noexcept {
    return autograd::Tensor(static_cast<double>(clifford_pinn_loss_exact(layer, collocation_pts, source_J, n_pts)), false);
}

} // namespace ucao::pinn
