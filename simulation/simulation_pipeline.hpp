#pragma once
// simulation/simulation_pipeline.hpp
// SimulationPipeline: OpenMP-параллельный симулятор N столов.
//
// Архитектура (план §3):
//   N параллельных столов, каждый — независимый BAEMAgent + HandSimulator.
//   Результаты агрегируются атомарно (lock-free).
//
// HPC Gate: инференс BAEM < 5% от времени тика симулятора.
//
// Тест 3 (Dynamic Regret 200K рук):
//   run_dynamic_regret_test() → vector<RegretPoint>
//   Проверяет: R_T ≤ C·(T·ε²_∞ + √T·σ_λ)

#include "../baem_v3.hpp"
#include "../gto_engine/cfr_engine.hpp"
#include "../baem_learning/hand_simulator.hpp"
#include "../baem_tracker/convergence_monitor.hpp"
#include <vector>
#include <atomic>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace baem_sim {

struct TableStats {
    double total_outcome_bb{0.0};
    int    hands_played{0};
    double running_variance{0.0};
    double running_mean{0.0};

    void record(float outcome) noexcept {
        ++hands_played;
        double delta = outcome - running_mean;
        running_mean += delta / hands_played;
        running_variance += delta * (outcome - running_mean);
        total_outcome_bb += outcome;
    }

    [[nodiscard]] double variance()         const noexcept {
        return (hands_played > 1) ? running_variance / (hands_played - 1) : 0.0;
    }
    [[nodiscard]] double winrate_bb_100()   const noexcept {
        return hands_played > 0 ? total_outcome_bb / hands_played * 100.0 : 0.0;
    }
};

struct SimulationStats {
    int    n_tables{0};
    int    total_hands{0};
    double aggregate_winrate_bb100{0.0};
    double aggregate_variance{0.0};
    double elapsed_seconds{0.0};
    double hands_per_second{0.0};
    double fraction_time_inference{0.0};
    std::vector<TableStats> per_table;

    void compute_aggregates() noexcept {
        total_hands = 0;
        double sw = 0.0, sv = 0.0;
        for (auto& t : per_table) {
            total_hands += t.hands_played;
            sw += t.winrate_bb_100();
            sv += t.variance();
        }
        n_tables = static_cast<int>(per_table.size());
        if (n_tables > 0) {
            aggregate_winrate_bb100 = sw / n_tables;
            aggregate_variance      = sv / n_tables;
        }
        hands_per_second = elapsed_seconds > 0
            ? total_hands / elapsed_seconds : 0.0;
    }

    void print() const noexcept {
        printf("=== SimulationStats ===\n");
        printf("  Tables:      %d\n",       n_tables);
        printf("  Total hands: %d\n",       total_hands);
        printf("  Winrate:     %.2f BB/100\n", aggregate_winrate_bb100);
        printf("  Variance:    %.2f BB²\n",    aggregate_variance);
        printf("  Throughput:  %.0f hands/s\n",hands_per_second);
        printf("  BAEM time:   %.2f%%\n",       fraction_time_inference * 100.0);
    }
};

struct OpponentProfile {
    float lambda{0.5f};
    float aggression{0.5f};
    float call_threshold{0.35f};
    float fold_threshold{0.15f};
    bool  drifts{false};
    int   drift_at_hand{5000};
    float lambda_after_drift{0.8f};
};

struct SimPipelineConfig {
    int   n_tables        = 4;
    int   hands_per_table = 500;
    int   omp_threads     = 1;
    bool  verbose         = false;
    bool  use_cfr_oracle  = true;
    OpponentProfile opp;
};

class SimulationPipeline {
public:
    using Config = SimPipelineConfig;

    explicit SimulationPipeline(Config cfg = {}) noexcept : cfg_(cfg) {
#ifdef _OPENMP
        omp_set_num_threads(cfg_.omp_threads);
#endif
    }

    [[nodiscard]] SimulationStats run() noexcept {
        SimulationStats stats;
        stats.per_table.resize(cfg_.n_tables);

        auto t0 = clk::now();
        std::atomic<long long> inf_ns_total{0};

#pragma omp parallel for schedule(dynamic,1) num_threads(cfg_.omp_threads) \
    default(none) shared(stats, inf_ns_total)
        for (int tbl = 0; tbl < cfg_.n_tables; ++tbl) {
            long long local_inf = 0;
            run_table(tbl, stats.per_table[tbl], local_inf);
            inf_ns_total.fetch_add(local_inf, std::memory_order_relaxed);
        }

        double elapsed = dur(clk::now() - t0);
        stats.elapsed_seconds = elapsed;
        stats.compute_aggregates();

        double total_ns = elapsed * 1e9;
        stats.fraction_time_inference =
            total_ns > 0 ? static_cast<double>(inf_ns_total.load()) / total_ns : 0.0;

        return stats;
    }

    // ── Тест 3: Dynamic Regret (T раздач, дрейфующий оппонент) ──────────
    struct RegretPoint {
        int   t{0};
        float regret_cumul{0.0f};
        float regret_bound{0.0f};  // теоретическая граница Formula 15
        float lambda_hat{0.0f};
        float epsilon_t{0.0f};
        bool  within_bound{false};
    };

    [[nodiscard]] std::vector<RegretPoint> run_dynamic_regret_test(
        int   T            = 50000,
        int   sample_every = 500,
        float sigma_lambda = 0.01f,
        float epsilon_inf  = 0.05f) noexcept
    {
        std::vector<RegretPoint> curve;
        curve.reserve(T / sample_every + 1);

        // Дрейфующая λ: меняется каждые 5K раздач
        static constexpr float LAMBDAS[] = {
            0.20f, 0.70f, 0.30f, 0.80f, 0.15f,
            0.65f, 0.40f, 0.90f, 0.25f, 0.60f};
        constexpr int DRIFT_PERIOD = 5000;

        baem::BAEMConfig acfg;
        acfg.use_simulator           = false;
        acfg.trainer_cfg.warmup_hands= 50;
        acfg.trainer_cfg.update_freq = 16;
        acfg.trainer_cfg.verbose     = false;

        baem::BAEMAgent agent(
            std::make_unique<gto::GTOOracleFromCFR>(
                gto::GTOOracleFromCFR::GameType::NLHEPreflop),
            std::make_unique<poker::HandEvaluator>(),
            acfg);

        baem_learning::HandSimulator sim(10000, 12345);

        poker::PublicState spub{};
        spub.num_players = 2;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> act_dist(1, 3);  // check/call/raise
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);

        float regret_cumul = 0.0f;
        float n_star_opt   = 300.0f;  // теоретический оптимум n* при известном λ

        for (int t = 1; t <= T; ++t) {
            // Текущая λ дрейфует
            float lam = LAMBDAS[((t-1) / DRIFT_PERIOD) % 10];

            // Симулируем раздачу
            auto result = sim.deal_and_run();
            float outcome = result.outcome_bb();

            // Масштабируем outcome по λ (более эксплуатируемый → больше выигрыш)
            outcome *= (1.0f + (1.0f - lam) * 0.5f);

            auto opp_action = static_cast<poker::ActionType>(act_dist(rng));
            spub.pot.total_bb100 = 150 + static_cast<int32_t>(ud(rng) * 600);

            agent.on_new_hand(spub);
            agent.on_opponent_action(spub, opp_action, outcome);
            agent.on_showdown(outcome);

            // Dynamic Regret: разница текущего n* vs оптимального
            // Оцениваем через |λ̂(t) - λ_true|² × масштаб (Formula 15)
            float lam_err = std::abs(agent.estimator().lambda_hat() - lam);
            float regret_step = lam_err * lam_err;  // квадратичная ошибка как proxy
            regret_cumul += regret_step;

            if (t % sample_every == 0) {
                float T_f    = static_cast<float>(t);
                // Граница R_T = O(T·ε²_∞ + √T·σ_λ) в единицах λ-ошибки²
                float bound  = T_f * epsilon_inf * epsilon_inf
                              + std::sqrt(T_f) * sigma_lambda;

                RegretPoint pt;
                pt.t            = t;
                pt.regret_cumul = regret_cumul;
                pt.regret_bound = bound;
                pt.lambda_hat   = agent.estimator().lambda_hat();
                pt.epsilon_t    = agent.estimator().epsilon_t();
                pt.within_bound = (regret_cumul <= bound * 10.0f); // 10x tolerance
                curve.push_back(pt);
            }
        }

        return curve;
    }

private:
    using clk = std::chrono::high_resolution_clock;
    static double dur(std::chrono::high_resolution_clock::duration d) noexcept {
        return std::chrono::duration<double>(d).count();
    }

    Config cfg_;

    void run_table(int idx, TableStats& stats, long long& inf_ns) noexcept {
        baem::BAEMConfig acfg;
        acfg.trainer_cfg.verbose      = false;
        acfg.trainer_cfg.warmup_hands = 30;
        acfg.trainer_cfg.update_freq  = 8;
        acfg.use_simulator            = false;

        baem::BAEMAgent agent(
            std::make_unique<gto::GTOOracleFromCFR>(
                cfg_.use_cfr_oracle
                ? gto::GTOOracleFromCFR::GameType::NLHEPreflop
                : gto::GTOOracleFromCFR::GameType::Kuhn),
            std::make_unique<poker::HandEvaluator>(),
            acfg);

        baem_learning::HandSimulator sim(10000,
            static_cast<uint64_t>(idx * 9999 + 13));

        poker::PublicState spub{};
        spub.num_players = 2;

        std::mt19937 rng(static_cast<uint32_t>(idx * 1234 + 5678));
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        std::uniform_int_distribution<int>    ad(1, 3);

        for (int h = 0; h < cfg_.hands_per_table; ++h) {
            auto result = sim.deal_and_run();

            // Initialize new hand (resets BeliefTracker)
            agent.on_new_hand(spub);

            auto t0  = clk::now();
            auto opp = static_cast<poker::ActionType>(ad(rng));
            agent.on_opponent_action(spub, opp, result.outcome_bb());

            std::array<float,5> ev = {-5.0f, 0.0f, 6.0f, 14.0f, 4.0f};
            agent.decide(spub, ev, ud(rng));
            inf_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          clk::now() - t0).count();

            agent.on_showdown(result.outcome_bb());
            stats.record(result.outcome_bb());
        }
    }
};

} // namespace baem_sim
