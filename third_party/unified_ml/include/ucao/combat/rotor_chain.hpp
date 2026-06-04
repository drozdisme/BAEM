#pragma once

#include "ucao/combat/motor_kinematics.hpp"

#include <array>
#include <cassert>

namespace ucao::combat {

/**
 * @brief Fixed-point rotor chain with periodic reorthogonalization.
 * @pre Depth > 0 and ReorthPeriod > 0.
 * @post Stores a chain of unit-like rotors initialized to identity.
 */
template <int N, int P, int Q, int Depth, int ReorthPeriod = 10>
struct FPRotorChain {
    using FPM = ucao::kernel::FPMultivector<N, P, Q>;

    std::array<FPM, Depth> rotors;

    FPRotorChain() noexcept {
        for (auto& rotor : rotors) {
            rotor.comps[0] = ucao::kernel::FP_SCALE;
        }
    }

    /**
     * @brief Apply the full chain as nested sandwiches.
     * @pre X is a valid multivector.
     * @post Returns R[Depth-1] ... R[0] X ~R[0] ... ~R[Depth-1].
     */
    [[nodiscard]] FPM apply_chain(const FPM& X) const noexcept {
        FPM tmp = X;
        for (int i = 0; i < Depth; ++i) {
            tmp = MotorKinematics<N, P, Q>::apply(rotors[i], tmp);
        }
        return tmp;
    }

    /**
     * @brief Update one rotor by right-multiplying an increment and re-normalizing periodically.
     * @pre 0 <= k < Depth.
     * @post rotors[k] = rotors[k] * increment and full reorthogonalization runs when step % ReorthPeriod == 0.
     */
    void update(int k, const FPM& increment, int step) noexcept {
        assert(k >= 0 && k < Depth);
        rotors[k] = ucao::kernel::fp_gp<N, P, Q>(rotors[k], increment);
        rotors[k].stabilize_rotor();
        if ((step % ReorthPeriod) == 0) {
            reorthogonalize_all();
        }
    }

    /**
     * @brief Re-normalize all chain rotors.
     * @pre The chain has been initialized.
     * @post Each rotor is reorthogonalized with 3 NR iterations.
     */
    void reorthogonalize_all() noexcept {
        for (auto& rotor : rotors) {
            rotor.reorthogonalize(3);
        }
    }

    /**
     * @brief Report the maximum normalization error across the chain.
     * @pre The chain is valid.
     * @post Returns max_i | ||R_i||^2 - 1 |.
     */
    [[nodiscard]] float max_rotor_error() const noexcept {
        float max_err = 0.0f;
        for (const auto& rotor : rotors) {
            const float err = rotor.rotor_error();
            if (err > max_err) {
                max_err = err;
            }
        }
        return max_err;
    }
};

} // namespace ucao::combat
