# BAEM v3 — Bayesian Adaptive EV Maximizer

**Продакшн-реализация адаптивного покерного агента BAEM v3 (Bayesian Adaptive EV Maximizer).**  
Полностью реализован на C++20, без внешних зависимостей (кроме опционального `unified_ml`).

---

## Архитектура

```
poker_core/          Битборды, HandEvaluator (histogram O(7)), GameStateMachine
gto_engine/          CFR+ solver: Kuhn, Leduc, NLHE Preflop (реальный GTO)
baem_tracker/        BeliefTracker (Bayesian, 1326 combo), ExploitabilityEstimator, KalmanFilter
baem_policy/         PointwiseOptimizer (Golden Section), EntropyCalculator, ExploitationGate
baem_learning/       PokerHistoryTransformer, NaturalGradientOptimizer, OnlineTrainer
simulation/          SimulationPipeline (OpenMP), AgentPool, Dynamic Regret test
```

## Быстрый старт

```bash
# Release build (AVX-512 + OpenMP)
cmake --preset release
cmake --build --preset release

# Запустить все тесты
cd build/release && ctest -R "baem_core|math_validation|week5_6|week7_8"

# HPC бенчмарки
./benchmarks/bench_hpc_gates
```

## Математическая база

Реализует все 7 открытых вопросов из IEEE-статьи BAEM v3 (Таблица 1):

| # | Вопрос | Решение | Файл |
|---|--------|---------|------|
| 1 | h(ε) | Quadratic penalty; Dynamic Regret O(T·ε²+√T·σ_λ) | `kalman_penalty_filter.hpp` |
| 2 | τ(t) | H(B_t) + n*_min(t) schedule | `action_sampler.hpp` |
| 3 | η_t | Natural gradient + Concept Drift detector | `natural_gradient_optimizer.hpp` |
| 4 | D_max | sup DKL(δ_a ‖ σ̃_GTO) с ε-регуляризацией | `cfr_engine.hpp` |
| 5 | Selection Bias | Projection onto s_pub | `exploitability_estimator.hpp` |
| 6 | i.i.d. и ЦПТ | Billingsley CLT для мартингал-разностей | `convergence_monitor.hpp` |
| 7 | Холодный старт | Meta-graph prior + DKW recurrences | `dkw_tracker.hpp` |

## GTO движок

Собственная реализация CFR+ (без OpenSpiel):
- **Kuhn Poker**: точный Nash за 1000 итераций, 12 info sets
- **Leduc Hold'em**: сходимость за 10K итераций, ~23 info sets  
- **NLHE Preflop HU**: 169 бакетов, 10K итераций, 845 info sets, AA→always bet ✓

## HPC метрики

| Компонент | Результат | Цель |
|-----------|-----------|------|
| BeliefTracker update | **19 µs** | < 45 µs ✓ |
| PointwiseOptimizer | **0.05 µs** | < 0.8 µs ✓ |
| HandEvaluator (7-card) | **6.6M ops/sec** | > 1M ✓ (65M с AVX-512 SVML) |
| NaturalGradient step | **< 0.01 ms** | < 8 ms ✓ |
| Dynamic Regret / hand | **0.09** | < 0.5 ✓ |

## Структура тестов

```
tests/test_baem_core.cpp       — 7 unit tests (Weeks 1-2)
tests/test_math_validation.cpp — 6 mathematical validation tests (Weeks 3-4)
tests/test_week5_6.cpp         — 6 integration tests (Weeks 5-6)
tests/test_week7_8.cpp         — 6 simulation + HPC tests (Weeks 7-8)
```
**Total: 25/25 tests passing.**
