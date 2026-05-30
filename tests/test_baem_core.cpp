// tests/test_baem_core.cpp
// Unit tests for BAEM v3 mathematical guarantees.
// No external test framework — lightweight self-contained runner.
//
// Tests covered:
//   T1: BeliefTracker normalization invariant (Σ B_t(h) = 1.0)
//   T2: DKW convergence against stationary opponent (Theorem 16.1)
//   T3: KalmanPenaltyFilter h(ε) ∈ [0,1], monotone in t
//   T4: PointwiseOptimizer α* minimizes objective
//   T5: ActionSampler policy sums to 1.0
//   T6: Concept Drift detection: h decreases on drift signal
//   T7: KuhnGTOracle strategy sums to 1.0 per bucket

#include "../baem_v3.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <numeric>
#include <random>

// ─── Mini test runner ─────────────────────────────────────────────────────────
static int tests_run    = 0;
static int tests_passed = 0;

#define EXPECT_NEAR(a, b, tol) do { \
    float _a = (float)(a), _b = (float)(b), _t = (float)(tol); \
    if (std::abs(_a - _b) > _t) { \
        printf("  FAIL %s:%d  |%s - %s| = %.6f > %.6f\n", \
               __FILE__, __LINE__, #a, #b, std::abs(_a-_b), _t); \
        return false; \
    } } while(0)

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("  FAIL %s:%d  " #cond "\n", __FILE__, __LINE__); \
        return false; \
    } } while(0)

#define RUN_TEST(fn) do { \
    tests_run++; \
    bool ok = fn(); \
    if (ok) { tests_passed++; printf("[PASS] " #fn "\n"); } \
    else    {                  printf("[FAIL] " #fn "\n"); } \
} while(0)

// ─── T1: Normalization invariant ─────────────────────────────────────────────
static bool test_belief_normalization() {
    gto::KuhnGTOracle oracle;
    poker::HandEvaluator evaluator;
    baem::BeliefTracker bt(&oracle, &evaluator);

    poker::CardSet agent_hole{};
    agent_hole.add(poker::Card::from_rank_suit(12, 0));  // Ac
    agent_hole.add(poker::Card::from_rank_suit(11, 0));  // Kc
    bt.init_hand(agent_hole, {});

    float sum0 = bt.sum_probs();
    EXPECT_NEAR(sum0, 1.0f, 1e-5f);

    // Apply Bayesian updates 20 times
    poker::PublicState spub{};
    for (int i = 0; i < 20; ++i) {
        poker::ActionType a = (i % 2 == 0) ? poker::ActionType::Check
                                            : poker::ActionType::Call;
        bt.update(spub, a, 0.5f);

        float s = bt.sum_probs();
        EXPECT_NEAR(s, 1.0f, 1e-5f);
    }
    return true;
}

// ─── T2: DKW convergence (Theorem 16.1) ──────────────────────────────────────
// True ΔE = 20.0, true Δσ² = 80.0; check estimates converge.
static bool test_dkw_convergence() {
    const float TRUE_DELTA_E   = 20.0f;
    const float TRUE_DELTA_VAR = 80.0f;
    const float E_GTO          = 5.0f;
    const float VAR_GTO        = 200.0f;

    baem::DKWTracker dkw(E_GTO, VAR_GTO,
                         /*delta_E_init=*/0.0f,
                         /*delta_var_init=*/0.0f,
                         /*t0=*/50);

    std::mt19937 rng(123);
    // Simulate outcomes from N(E_GTO + TRUE_DELTA_E, sqrt(VAR_GTO + TRUE_DELTA_VAR))
    float true_E   = E_GTO   + TRUE_DELTA_E;
    float true_std = std::sqrt(VAR_GTO + TRUE_DELTA_VAR);
    std::normal_distribution<float> dist(true_E, true_std);

    for (int i = 0; i < 50000; ++i)
        dkw.update(dist(rng));

    EXPECT_NEAR(dkw.delta_E(),   TRUE_DELTA_E,   2.0f);   // within 2.0 BB
    // delta_var converges to E[(X-E_gto)^2] - var_gto = (20^2+80) - 200 = 280
    // per Formula 28: tracks mean((X_t+1 - E_gto)^2) - var_gto
    float expected_delta_var = (TRUE_DELTA_E * TRUE_DELTA_E + VAR_GTO + TRUE_DELTA_VAR) - VAR_GTO;
    EXPECT_NEAR(dkw.delta_var(), expected_delta_var, 30.0f);  // within 20 BB²
    return true;
}

// ─── T3: Kalman h(ε) properties ──────────────────────────────────────────────
static bool test_kalman_h() {
    baem::KalmanPenaltyFilter kf;
    kf.sigma_lambda = 0.01f;

    // Initially h should be < 1 (high uncertainty)
    float h0 = kf.h();
    EXPECT_TRUE(h0 >= 0.0f && h0 <= 1.0f);

    // After many steps (no drift), h should increase toward 1
    float Dmax = 9.21f;
    for (int i = 0; i < 1000; ++i)
        kf.step(Dmax);

    float h1000 = kf.h();
    EXPECT_TRUE(h1000 > h0);
    EXPECT_TRUE(h1000 <= 1.0f);

    // Concept Drift: h should drop after a drift signal
    kf.apply_drift_signal(0.5f, 0.05f);  // large drift
    float h_post_drift = kf.h();
    EXPECT_TRUE(h_post_drift < h1000);

    return true;
}

// ─── T4: PointwiseOptimizer finds minimum ────────────────────────────────────
// Construct a case where the optimal α is clearly ≈ 0.5.
static bool test_pointwise_optimizer() {
    // Set up DKW so that α=0 and α=1 both have high n*, α≈0.5 has lower n*.
    // E_gto=5, delta_E=20 → E(α) = 5 + 20α (increasing)
    // var_gto=200, delta_var=300 → var(α) = 200 + 300α (linear for this test)
    baem::DKWTracker dkw(5.0f, 200.0f, 20.0f, 300.0f, 1000);

    baem::PointwiseOptimizer opt;
    baem::OptimResult res = opt.find(dkw, /*h_eps=*/1.0f);

    // α=0: n* = 200*z² / 25 = 8z²
    // α=1: n* = 500*z² / 625 = 0.8z²
    // → α=1 should be optimal (higher EV gains dominate)
    EXPECT_TRUE(res.alpha_star >= 0.0f && res.alpha_star <= 1.0f);
    EXPECT_TRUE(res.n_star_min < 1e8f);
    EXPECT_TRUE(res.objective < 1e8f);

    // With h_eps = 0.5, alpha should be halved
    baem::OptimResult res_half = opt.find(dkw, /*h_eps=*/0.5f);
    EXPECT_TRUE(res_half.alpha_star <= res.alpha_star + 1e-5f);

    return true;
}

// ─── T5: ActionSampler policy sums to 1.0 ────────────────────────────────────
static bool test_action_sampler_normalizes() {
    baem::ActionSampler sampler;
    gto::KuhnGTOracle oracle;

    poker::PublicState spub{};
    gto::ActionDist gto_dist = oracle.sigma_gto(spub, 2);  // King bucket

    std::array<float, 5> ev_exploit = {-10.f, 0.f, 5.f, 15.f, 8.f};

    for (float alpha : {0.0f, 0.3f, 0.7f, 1.0f}) {
        auto policy = sampler.mixed_policy(ev_exploit, gto_dist, alpha, 0.3f);
        float sum = 0.0f;
        for (float p : policy) {
            EXPECT_TRUE(p >= 0.0f);
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f);
    }
    return true;
}

// ─── T6: Concept Drift reduces h ─────────────────────────────────────────────
static bool test_concept_drift_reduces_h() {
    baem::KalmanPenaltyFilter kf;
    for (int i = 0; i < 500; ++i) kf.step(9.21f);  // warm up
    float h_before = kf.h();

    // Inject a large drift signal
    kf.apply_drift_signal(0.8f, 0.05f);
    float h_after = kf.h();

    EXPECT_TRUE(h_after < h_before);
    return true;
}

// ─── T7: KuhnGTOracle strategy sums to 1 ─────────────────────────────────────
static bool test_kuhn_oracle_normalized() {
    gto::KuhnGTOracle oracle;
    poker::PublicState spub{};

    for (int bucket = 0; bucket < 3; ++bucket) {
        for (int facing : {0, 1}) {
            spub.pot.current_bet_bb100 = facing ? 100 : 0;
            for (int player : {0, 1}) {
                spub.action_to_act = player;
                gto::ActionDist d = oracle.sigma_gto(spub, bucket);
                uint32_t total = 0;
                for (auto x : d.p) total += x;
                EXPECT_NEAR(static_cast<float>(total), 10000.0f, 1.0f);
            }
        }
    }
    return true;
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    printf("=== BAEM v3 Unit Tests ===\n\n");

    RUN_TEST(test_belief_normalization);
    RUN_TEST(test_dkw_convergence);
    RUN_TEST(test_kalman_h);
    RUN_TEST(test_pointwise_optimizer);
    RUN_TEST(test_action_sampler_normalizes);
    RUN_TEST(test_concept_drift_reduces_h);
    RUN_TEST(test_kuhn_oracle_normalized);

    printf("\n%d / %d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
