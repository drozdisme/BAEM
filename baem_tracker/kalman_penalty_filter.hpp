#pragma once
// baem_tracker/kalman_penalty_filter.hpp
// Implements the dynamic penalty h(ε) and Kalman filter for λ tracking.
//
// Formulas (12)–(14) from BAEM v3:
//
//   dε²/dt = 2σ²_λ - ε⁴(t) / σ²_obs(t)     [Kalman variance ODE]
//   h(ε(t)) = 1 - C₁ε(t) - C₂ε²(t)           [quadratic penalty]
//
// Steady-state error: ε_∞ = √(σ_λ · σ_obs)
//
// Dynamic Regret bound (Formula 15):
//   R_T = O(T·ε²_∞ + √T·σ_λ)
//
// Concept Drift handling:
//   When ||Δσ̂_opp||₁ spikes, σ²_λ estimate jumps →
//   filter broadens confidence → h(ε) decreases → α*(t) pulled toward GTO.
//
// Discrete approximation: Euler step with dt = 1/n*(t).

#include <cmath>
#include <algorithm>
#include <cassert>

namespace baem {

struct KalmanPenaltyFilter {
    // ─── Parameters ──────────────────────────────────────────────────────────
    float sigma_lambda{0.01f};    // drift volatility of opponent's λ
    float sigma_obs_base{0.5f};   // baseline observation noise
    float C1{0.5f};               // computed from derivatives of n*(α*,λ̂)
    float C2{0.25f};               // computed from second derivatives

    // ─── State ───────────────────────────────────────────────────────────────
    float epsilon2{1.0f};        // current variance ε²(t) — starts high (no info)
    int   t{0};                  // step counter (number of hands played)

    // Compute σ²_obs(t) = Dmax / t (Proposition after Formula 12)
    [[nodiscard]] float sigma_obs_sq(float Dmax) const noexcept {
        if (t == 0) return Dmax;
        return Dmax / static_cast<float>(t) + sigma_obs_base * sigma_obs_base;
    }

    // Euler step for the Kalman variance ODE (Formula 12)
    // dt ≈ 1 hand (n*_min normalization applied externally if desired)
    void step(float Dmax, float drift_boost = 1.0f) noexcept {
        t++;
        float s_obs2 = sigma_obs_sq(Dmax);
        float s_lam2 = sigma_lambda * sigma_lambda * drift_boost * drift_boost;

        // dε²/dt = 2σ²_λ - ε⁴ / σ²_obs
        float eps4  = epsilon2 * epsilon2;
        float d_eps2 = 2.0f * s_lam2 - eps4 / s_obs2;

        epsilon2 += d_eps2;  // dt = 1 step
        epsilon2  = std::max(1e-6f, epsilon2);  // numerical floor

        // Recompute C1, C2 from current state
        // Conservative defaults from Taylor expansion analysis:
        // C1 = 1/(2·λ̂_expected), C2 = C1²
        // In production: compute exactly from ∂n*/∂λ at current α*, λ̂.
        C1 = 0.5f;
        C2 = 0.25f;
    }

    // Concept Drift: if ||Δσ_opp||₁ exceeds threshold, inflate σ²_λ
    void apply_drift_signal(float delta_l1, float threshold = 0.05f) noexcept {
        if (delta_l1 > threshold) {
            // Boost: inflate variance proportionally to detected shift
            float boost = 1.0f + delta_l1 / threshold;
            epsilon2 *= boost * boost;
            // Also partially reset the accumulated time (lose trust in history)
            t = std::max(1, static_cast<int>(t * 0.3f));
        }
    }

    // ─── h(ε(t)) — quadratic penalty (Formula 13) ───────────────────────────
    [[nodiscard]] float h() const noexcept {
        float eps = std::sqrt(epsilon2);
        float val = 1.0f - C1 * eps - C2 * epsilon2;
        return std::clamp(val, 0.05f, 1.0f);  // floor at 5% to avoid zero
    }

    // ─── Steady-state error ε_∞ (Remark after Formula 12) ──────────────────
    [[nodiscard]] float epsilon_inf(float Dmax) const noexcept {
        // ε_∞ = √(σ_λ · σ_obs)
        return std::sqrt(sigma_lambda * sigma_obs_base);
    }

    // ─── Dynamic Regret contribution at current step ─────────────────────────
    [[nodiscard]] float regret_term() const noexcept {
        return epsilon2;  // proportional to ε²(t)
    }

    // ─── Reset (new session / new opponent) ─────────────────────────────────
    void reset() noexcept {
        epsilon2 = 1.0f;
        t = 0;
    }
};

} // namespace baem
