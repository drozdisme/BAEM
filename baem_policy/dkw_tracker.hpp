#pragma once
// baem_policy/dkw_tracker.hpp
// Implements Formulas (27)–(28) from BAEM v3 (§XVI: Cold Start).
//
// Robbins-Monro (DKW) recurrences:
//   ΔÊ(t+1) = ΔÊ(t) + γ_t · (X_{t+1} - Ê_GTO - ΔÊ(t))
//   Δσ̂²(t+1)= Δσ̂²(t) + γ_t · ((X_{t+1}-Ê_GTO)² - σ²_GTO - Δσ̂²(t))
//
// Learning rate: γ_t = 1 / (t₀ + t)  → Σγ_t = ∞, Σγ²_t < ∞ ✓
//
// Cold start (Formula 26): initialize from meta-graph D
//   ΔÊ(0)  = E_{θ₀~P(D)}[ΔE(θ₀)]
//   Δσ̂²(0) = E_{θ₀~P(D)}[Δσ²(θ₀)]
//
// Theorem 16.1 guarantees unbiased estimates from t=0 onwards.

#include <cmath>
#include <algorithm>
#include <cassert>

namespace baem {

class DKWTracker {
public:
    // Constructor: provide GTO baseline values and cold-start prior.
    // delta_E_init, delta_var_init: meta-graph prior (Section XVI)
    explicit DKWTracker(
        float E_gto_prior      = 0.0f,   // Ê_GTO (baseline EV)
        float var_gto_prior    = 250.0f, // σ²_GTO baseline (in BB²×100)
        float delta_E_init     = 5.0f,   // prior on ΔÊ (from meta-graph)
        float delta_var_init   = 50.0f,  // prior on Δσ̂² (from meta-graph)
        int   t0               = 100     // Robbins-Monro offset (controls initial LR)
    ) noexcept
        : E_gto_(E_gto_prior)
        , var_gto_(var_gto_prior)
        , delta_E_(delta_E_init)
        , delta_var_(delta_var_init)
        , t0_(t0)
    {}

    // Update after observing outcome X_{t+1} (in BB×100).
    void update(float X_next) noexcept {
        t_++;
        float gamma = 1.0f / static_cast<float>(t0_ + t_);

        // Formula (27): update ΔÊ
        float residual_E   = X_next - E_gto_ - delta_E_;
        delta_E_  += gamma * residual_E;

        // Formula (28): update Δσ̂²
        float centered     = X_next - E_gto_;
        float residual_var = (centered * centered) - var_gto_ - delta_var_;
        delta_var_ += gamma * residual_var;
        delta_var_  = std::max(0.0f, delta_var_);  // variance must be non-negative

        // Running statistics for diagnostics
        running_mean_  += (X_next - running_mean_) / static_cast<float>(t_);
        float var_incr = (X_next - running_mean_);
        running_var_  += (var_incr * var_incr - running_var_) / static_cast<float>(t_);
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    // Total estimated EV for exploitative strategy at given α
    [[nodiscard]] float E_hat(float alpha) const noexcept {
        return E_gto_ + alpha * delta_E_;
    }

    // Total estimated variance at given α
    [[nodiscard]] float var_hat(float alpha) const noexcept {
        return var_gto_ + alpha * delta_var_;
    }

    // Raw incremental quantities (for PointwiseOptimizer)
    [[nodiscard]] float delta_E()   const noexcept { return delta_E_;   }
    [[nodiscard]] float delta_var() const noexcept { return delta_var_; }
    [[nodiscard]] float E_gto()     const noexcept { return E_gto_;     }
    [[nodiscard]] float var_gto()   const noexcept { return var_gto_;   }
    [[nodiscard]] int   num_hands() const noexcept { return t_;         }

    // n*_min estimate (Formula 2 / Lemma 1.3):
    // n*(α) ≍ σ²(α) · z² / E[X(α)]²
    [[nodiscard]] float n_star_min(float alpha, float z_delta = 2.576f) const noexcept {
        float E   = E_hat(alpha);
        float var = var_hat(alpha);
        if (std::abs(E) < 1e-6f) return 1e9f;  // undefined / infinite
        return var * z_delta * z_delta / (E * E);
    }

    // Running sample mean and variance (for diagnostics / CLT validation)
    [[nodiscard]] float running_mean() const noexcept { return running_mean_; }
    [[nodiscard]] float running_var()  const noexcept { return running_var_;  }

    void reset() noexcept {
        t_           = 0;
        running_mean_= 0.0f;
        running_var_ = 0.0f;
        // Preserve priors (cold-start initialization)
    }

private:
    float E_gto_{};
    float var_gto_{};
    float delta_E_{};
    float delta_var_{};
    int   t0_{100};
    int   t_{0};

    float running_mean_{0.0f};
    float running_var_{0.0f};
};

} // namespace baem
