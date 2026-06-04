#pragma once

#include "autograd/autograd.h"
#include "core/random.hpp"
#include "ucao/kernel/dual_mv.hpp"
#include "ucao/kernel/vector_derivative.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace ucao::pinn {

/**
 * @brief Clifford-equivariant field layer built from rotor sandwiches.
 * @pre HiddenDim > 0 and N = P + Q.
 * @post Stores HiddenDim rotor weights and bias multivectors.
 */
template <int N, int P, int Q, int HiddenDim = 64>
struct CliffordFieldLayer {
    static_assert(P + Q == N, "ucao: algebra dimensions must sum to N");
    static_assert(N <= 5, "ucao: N > 5 requires 64-wide SIMD, not implemented");

    static constexpr int D = 1 << N;

    alignas(64) float weights[HiddenDim][D]{};
    alignas(64) float bias[HiddenDim][D]{};

    explicit CliffordFieldLayer(unsigned seed = 42) {
        core::RNG rng(seed);
        const double scale = 1.0 / std::sqrt(static_cast<double>(HiddenDim));
        for (int h = 0; h < HiddenDim; ++h) {
            for (int i = 0; i < D; ++i) {
                weights[h][i] = static_cast<float>(rng.uniform_real(-0.5, 0.5) * scale);
                bias[h][i] = static_cast<float>(rng.uniform_real(-0.5, 0.5) * scale);
            }
            for (int i = 0; i < D; ++i) {
                if ((ucao::kernel::detail::grade_of<N>(i) & 1) != 0) {
                    weights[h][i] = 0.0f;
                }
            }
            if (std::fabs(weights[h][0]) < 1e-4f) {
                weights[h][0] += 1.0f;
            }
            clifford_norm_activate(weights[h], D);
        }
        weight_tensors_.reserve(HiddenDim);
        bias_tensors_.reserve(HiddenDim);
        for (int h = 0; h < HiddenDim; ++h) {
            std::vector<double> w(static_cast<std::size_t>(D));
            std::vector<double> b(static_cast<std::size_t>(D));
            for (int i = 0; i < D; ++i) {
                w[i] = static_cast<double>(weights[h][i]);
                b[i] = static_cast<double>(bias[h][i]);
            }
            weight_tensors_.emplace_back(std::move(w), std::vector<std::size_t>{static_cast<std::size_t>(D)}, true);
            bias_tensors_.emplace_back(std::move(b), std::vector<std::size_t>{static_cast<std::size_t>(D)}, true);
        }
    }

    /**
     * @brief Forward pass from coordinates to a normalized multivector field value.
     * @pre coord points to N scalar coordinates and out points to D floats.
     * @post out is the normalized hidden multivector sum.
     */
    void forward(const float* coord, float* out) const noexcept {
        alignas(64) float x_mv[D]{};
        alignas(64) float accum[D]{};
        for (int i = 0; i < N; ++i) {
            x_mv[1 << i] = coord[i];
        }
        for (int h = 0; h < HiddenDim; ++h) {
            const auto tmp = ucao::kernel::DualMV<N, P, Q>::sandwich(weights[h], ucao::kernel::DualMV<N, P, Q>::from_real(x_mv));
            for (int i = 0; i < D; ++i) {
                accum[i] += tmp.real[i] + bias[h][i];
            }
        }
        std::memcpy(out, accum, sizeof(accum));
        clifford_norm_activate(out, D);
    }

    /**
     * @brief Dual forward path for exact coordinate derivatives.
     * @pre x is a dual multivector whose dual part seeds one coordinate direction.
     * @post Returns the same field computation as forward(), but in dual algebra.
     */
    [[nodiscard]] ucao::kernel::DualMV<N, P, Q> forward_dual(const ucao::kernel::DualMV<N, P, Q>& x) const noexcept {
        ucao::kernel::DualMV<N, P, Q> accum;
        for (int h = 0; h < HiddenDim; ++h) {
            const auto tmp = ucao::kernel::DualMV<N, P, Q>::sandwich(weights[h], x);
            for (int i = 0; i < D; ++i) {
                accum.real[i] += tmp.real[i] + bias[h][i];
                accum.dual[i] += tmp.dual[i];
            }
        }
        return accum.clifford_norm();
    }

    /**
     * @brief Return optimizer-facing parameter tensors.
     * @pre The layer has been constructed.
     * @post Returns stable pointers to cached tensor copies of weights and biases.
     */
    [[nodiscard]] std::vector<autograd::Tensor*> parameters() {
        std::vector<autograd::Tensor*> params;
        params.reserve(static_cast<std::size_t>(HiddenDim) * 2);
        for (int h = 0; h < HiddenDim; ++h) {
            params.push_back(&weight_tensors_[h]);
            params.push_back(&bias_tensors_[h]);
        }
        return params;
    }

    /**
     * @brief Synchronize float buffers from optimizer-owned tensor copies.
     * @pre weight_tensors_ and bias_tensors_ were mutated externally.
     * @post weights and bias arrays mirror tensor data exactly.
     */
    void sync_from_tensors() noexcept {
        for (int h = 0; h < HiddenDim; ++h) {
            const auto w = weight_tensors_[h].data();
            const auto b = bias_tensors_[h].data();
            for (int i = 0; i < D; ++i) {
                weights[h][i] = static_cast<float>(w[i]);
                bias[h][i] = static_cast<float>(b[i]);
            }
        }
    }

private:
    std::vector<autograd::Tensor> weight_tensors_;
    std::vector<autograd::Tensor> bias_tensors_;

    /**
     * @brief Normalize a Clifford multivector by its reverse metric norm.
     * @pre m points to d coefficients.
     * @post m is divided by sqrt(|sum rev_sign(i) * m_i^2| + eps).
     */
    static void clifford_norm_activate(float* m, int d) noexcept {
        constexpr float eps = 1e-8f;
        float n2 = 0.0f;
        for (int i = 0; i < d; ++i) {
            n2 += static_cast<float>(ucao::kernel::detail::reversion_sign<N>(i)) * m[i] * m[i];
        }
        const float inv = 1.0f / std::sqrt(std::fabs(n2) + eps);
        for (int i = 0; i < d; ++i) {
            m[i] *= inv;
        }
    }
};

} // namespace ucao::pinn
