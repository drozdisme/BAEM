#pragma once

#include "ucao/kernel/fp_multivector.hpp"

namespace ucao::combat {

using fp64 = ucao::kernel::fp64;

/**
 * @brief Deterministic motor kinematics in fixed-point Clifford algebra.
 * @pre N = P + Q and operands belong to the same algebra.
 * @post Provides apply, compose and normalized interpolation on motors.
 */
template <int N, int P, int Q>
struct MotorKinematics {
    static_assert(ucao::kernel::GPSignTable<N, P, Q>::D == (1 << N), "ucao: algebra dimensions must sum to N");

    using FPM = ucao::kernel::FPMultivector<N, P, Q>;

    /**
     * @brief Apply a motor sandwich M X ~M.
     * @pre M and X are valid fixed-point multivectors.
     * @post Returns fp_gp(fp_gp(M, X), rev(M)).
     */
    [[nodiscard]] static FPM apply(const FPM& M, const FPM& X) noexcept {
        const FPM tmp = ucao::kernel::fp_gp<N, P, Q>(M, X);
        const FPM revM = M.reversed();
        return ucao::kernel::fp_gp<N, P, Q>(tmp, revM);
    }

    /**
     * @brief Compose two motors as M2 * M1.
     * @pre M1 and M2 are valid motors.
     * @post Returns the left-to-right composition motor.
     */
    [[nodiscard]] static FPM compose(const FPM& M1, const FPM& M2) noexcept {
        return ucao::kernel::fp_gp<N, P, Q>(M2, M1);
    }

    /**
     * @brief Linearly interpolate two motors in Q30 and renormalize.
     * @pre t_q30 is in [0, FP_SCALE].
     * @post Returns a reorthogonalized interpolated motor.
     */
    [[nodiscard]] static FPM lerp_normalized(const FPM& M1, const FPM& M2, fp64 t_q30) noexcept {
        FPM out;
        for (int i = 0; i < FPM::D; ++i) {
            const __int128 delta = static_cast<__int128>(M2.comps[i]) - static_cast<__int128>(M1.comps[i]);
            out.comps[i] = M1.comps[i] + static_cast<fp64>((delta * t_q30) >> ucao::kernel::FP_LOG2);
        }
        out.reorthogonalize(3);
        return out;
    }
};

} // namespace ucao::combat
