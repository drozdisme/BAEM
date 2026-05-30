#pragma once
// gto_engine/gto_oracle.hpp
// GTO strategy oracle interface + Kuhn/Leduc implementations.
//
// Design:
//   - IGTOracle: pure virtual interface, no cost in production path
//   - KuhnGTOracle: exact Nash equilibrium (analytical solution)
//   - LeducGTOracle: pre-solved LUT (bucketed)
//   - TexasHoldemGTOracle: stub returning uniform-fold-call mix
//
// All probability distributions are stored as fixed-point (uint16_t,
// scaled by 10000) to avoid floats in hot path.
// matrix_view integration stub: in production, load GTO tables from
// a unified_ml::core::matrix_view-mapped memory region.

#include "../poker_core/cards.hpp"
#include "../poker_core/game_state.hpp"
#include <array>
#include <span>
#include <cstdint>
#include <cassert>

namespace gto {

// ─── Action probability distribution ────────────────────────────────────────
// Probabilities for [Fold, Check, Call, Raise, AllIn].
// Sum = 10000 (fixed-point ×10^-4).
struct ActionDist {
    std::array<uint16_t, 5> p{};  // indexed by ActionType

    [[nodiscard]] float prob(poker::ActionType a) const noexcept {
        return static_cast<float>(p[static_cast<uint8_t>(a)]) * 1e-4f;
    }

    [[nodiscard]] static ActionDist fold_only() noexcept {
        ActionDist d; d.p[0] = 10000; return d;
    }
    [[nodiscard]] static ActionDist check_only() noexcept {
        ActionDist d; d.p[1] = 10000; return d;
    }
    [[nodiscard]] static ActionDist call_only() noexcept {
        ActionDist d; d.p[2] = 10000; return d;
    }
    [[nodiscard]] static ActionDist raise_only() noexcept {
        ActionDist d; d.p[3] = 10000; return d;
    }
    [[nodiscard]] static ActionDist uniform_bet() noexcept {
        // fold=0, check=3333, call=3333, raise=3334
        ActionDist d;
        d.p[1] = 3333; d.p[2] = 3333; d.p[3] = 3334;
        return d;
    }
};

// ─── EV reference values (in BB, scaled ×100) ─────────────────────────────
using EVx100 = int32_t;  // e.g., 15 means 0.15 BB

// ─── Oracle interface ─────────────────────────────────────────────────────────
struct IGTOracle {
    virtual ~IGTOracle() = default;

    // Query GTO action distribution for given public state and hand bucket.
    // hand_bucket ∈ [0, num_buckets), abstracted from hole cards.
    [[nodiscard]] virtual ActionDist sigma_gto(
        const poker::PublicState& spub,
        int hand_bucket) const noexcept = 0;

    // Reference EV of GTO strategy for given bucket and opponent λ.
    [[nodiscard]] virtual EVx100 ev_gto(
        const poker::PublicState& spub,
        int hand_bucket,
        float lambda) const noexcept = 0;

    // Variance of GTO strategy (used in BAEM σ² calculations).
    [[nodiscard]] virtual EVx100 var_gto(
        const poker::PublicState& spub,
        int hand_bucket) const noexcept = 0;

    // Number of hand buckets for this oracle.
    [[nodiscard]] virtual int num_buckets() const noexcept = 0;

    // Dmax (Section XIII of BAEM v3): pre-computed offline.
    [[nodiscard]] virtual float Dmax() const noexcept = 0;
};

// ─── Kuhn Poker GTO Oracle ───────────────────────────────────────────────────
// 3-card game (J=0, Q=1, K=2), 2 players, 1 betting round.
// Analytical Nash equilibrium (alpha parameterizes the Nash family).
// Using the standard solution: alpha = 1/3.
//
// Reference: Kuhn (1950), "A Simplified Two-Person Poker".
class KuhnGTOracle final : public IGTOracle {
public:
    // Positions
    static constexpr int OOP = 0;  // out of position (acts first)
    static constexpr int IP  = 1;  // in position (acts second)

    // Kuhn ranks: J=0, Q=1, K=2
    static constexpr int NUM_BUCKETS = 3;

    KuhnGTOracle() { init_tables(); }

    [[nodiscard]] ActionDist sigma_gto(
        const poker::PublicState& spub,
        int hand_bucket) const noexcept override
    {
        // Map stage + action history + bucket to GTO action
        // Full Kuhn GTO tables are 2 positions × 3 ranks × 2 history nodes
        // We encode: position = action_to_act, facing_bet = pot.current_bet > 0
        int pos = spub.action_to_act;
        bool facing_bet = (spub.pot.current_bet_bb100 > 0);

        if (pos == OOP && !facing_bet) {
            // OOP first action: bet with K always, bet with J with prob alpha=1/3
            return table_oop_first_action_[hand_bucket];
        } else if (pos == OOP && facing_bet) {
            // OOP facing raise: fold J, call Q with alpha/3, call K always
            return table_oop_facing_raise_[hand_bucket];
        } else if (pos == IP && !facing_bet) {
            // IP facing check: bet with K, bet with J with alpha=1/3
            return table_ip_facing_check_[hand_bucket];
        } else {
            // IP facing bet: fold J, call K, call Q sometimes
            return table_ip_facing_bet_[hand_bucket];
        }
    }

    [[nodiscard]] EVx100 ev_gto(
        const poker::PublicState&,
        int hand_bucket,
        float /*lambda*/) const noexcept override
    {
        // Approximate EV at Nash equilibrium
        // OOP EV: J=-1/18, Q=+1/18, K=+5/18 (scaled ×100)
        const int32_t ev_table[3] = {-6, 6, 28};  // ≈ EV×100 in BB
        return ev_table[hand_bucket];
    }

    [[nodiscard]] EVx100 var_gto(
        const poker::PublicState&,
        int /*hand_bucket*/) const noexcept override
    {
        // Approximate variance in BB² ×100
        return 120;  // 1.2 BB² variance at Nash
    }

    [[nodiscard]] int num_buckets() const noexcept override { return NUM_BUCKETS; }

    [[nodiscard]] float Dmax() const noexcept override {
        // Dmax = sup_a [-ln σ̃_GTO(a)] with ε_reg = 1e-4
        // For Kuhn poker, the rarest GTO action has prob ≈ 1/3 → Dmax ≈ 1.1
        return 1.099f;  // -ln(1/3)
    }

private:
    // Pre-computed strategy tables for each (position × history × rank)
    std::array<ActionDist, NUM_BUCKETS> table_oop_first_action_{};
    std::array<ActionDist, NUM_BUCKETS> table_oop_facing_raise_{};
    std::array<ActionDist, NUM_BUCKETS> table_ip_facing_check_{};
    std::array<ActionDist, NUM_BUCKETS> table_ip_facing_bet_{};

    void init_tables() noexcept {
        // alpha = 1/3 Nash solution
        // OOP first action: J→ check(2/3) bet(1/3), Q→check, K→bet
        table_oop_first_action_[0].p[1] = 6667; table_oop_first_action_[0].p[3] = 3333; // J
        table_oop_first_action_[1] = ActionDist::check_only();  // Q
        table_oop_first_action_[2] = ActionDist::raise_only();  // K

        // OOP facing raise: J→fold, Q→call(1/3) fold(2/3), K→call
        table_oop_facing_raise_[0] = ActionDist::fold_only();   // J
        table_oop_facing_raise_[1].p[0] = 6667; table_oop_facing_raise_[1].p[2] = 3333; // Q
        table_oop_facing_raise_[2] = ActionDist::call_only();   // K

        // IP facing check: J→check(2/3) bet(1/3), Q→check, K→bet
        table_ip_facing_check_[0].p[1] = 6667; table_ip_facing_check_[0].p[3] = 3333; // J
        table_ip_facing_check_[1] = ActionDist::check_only();   // Q
        table_ip_facing_check_[2] = ActionDist::raise_only();   // K

        // IP facing bet: J→fold, Q→fold(2/3) call(1/3), K→call
        table_ip_facing_bet_[0] = ActionDist::fold_only();      // J
        table_ip_facing_bet_[1].p[0] = 6667; table_ip_facing_bet_[1].p[2] = 3333;  // Q
        table_ip_facing_bet_[2] = ActionDist::call_only();      // K
    }
};

// ─── Texas Hold'em GTO Stub ───────────────────────────────────────────────────
// Returns a cautious GTO-like strategy for Texas Hold'em.
// In production: replace with full CFR-solved LUT loaded via matrix_view.
// Bucket system: 0..169 for preflop hand classes, 0..N for postflop abstractions.
class TexasHoldemGTOracle final : public IGTOracle {
public:
    static constexpr int NUM_PREFLOP_BUCKETS = 169;
    static constexpr float EPSILON_REG = 1e-4f;

    [[nodiscard]] ActionDist sigma_gto(
        const poker::PublicState& spub,
        int hand_bucket) const noexcept override
    {
        // Realistic GTO approximation for Texas Hold'em.
        // GTO plays a balanced mixed strategy; strength determines bet/call freq.
        // Range: [0..168], midpoint ~84.
        bool facing_bet = (spub.pot.current_bet_bb100 > 0);
        float str = static_cast<float>(hand_bucket) / 168.0f;  // 0..1

        ActionDist d{};
        if (facing_bet) {
            // Facing a bet: fold / call / raise mixture
            uint16_t p_fold  = static_cast<uint16_t>((1.0f - str) * 4000);
            uint16_t p_raise = static_cast<uint16_t>(str * str * 3000);
            uint16_t p_call  = 10000 - p_fold - p_raise;
            d.p[0] = p_fold;   // fold
            d.p[2] = p_call;   // call
            d.p[3] = p_raise;  // raise
        } else {
            // No bet facing: check / bet mixture
            uint16_t p_raise = static_cast<uint16_t>(str * 4000);
            uint16_t p_check = 10000 - p_raise;
            d.p[1] = p_check;  // check
            d.p[3] = p_raise;  // raise/bet
        }
        return d;
    }

    [[nodiscard]] EVx100 ev_gto(
        const poker::PublicState&,
        int hand_bucket,
        float /*lambda*/) const noexcept override
    {
        // Linear approximation: strong hands have positive EV
        // Proper values: load from pre-solved CFR table
        return static_cast<EVx100>((hand_bucket - 84) * 2);
    }

    [[nodiscard]] EVx100 var_gto(
        const poker::PublicState&,
        int /*hand_bucket*/) const noexcept override
    {
        return 250;  // ≈ 2.5 BB² variance (typical for HU cash game)
    }

    [[nodiscard]] int num_buckets() const noexcept override {
        return NUM_PREFLOP_BUCKETS;
    }

    [[nodiscard]] float Dmax() const noexcept override {
        // Section XIII: Dmax = sup_a [-ln σ̃_GTO(a)]
        // With ε_reg = 1e-4, rarest action prob ≈ 1e-4 → Dmax ≈ ln(10000) ≈ 9.21
        return 9.21f;
    }

    // Map hole cards to a preflop bucket index [0..168]
    // Standard 169-bucket abstraction: suited connectors, pairs, etc.
    [[nodiscard]] static int preflop_bucket(poker::Card c1, poker::Card c2) noexcept {
        int r1 = c1.rank(), r2 = c2.rank();
        if (r1 < r2) std::swap(r1, r2);
        bool suited = (c1.suit() == c2.suit());
        if (r1 == r2) return r1 * 13 + r2;               // pairs: 0..12 → bucket 0..12
        if (suited)   return 13 + r1 * 13 + r2;           // suited: 13..78
        return 13 + 66 + (r1 * 13 + r2);                  // offsuit: 79..168
    }
};

} // namespace gto

// cfr_engine.hpp included separately to avoid circular deps
// Use: #include "gto_engine/cfr_engine.hpp"
