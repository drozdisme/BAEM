# BAEM v3 Performance Report

**Дата:** 2026-05-27  
**Платформа:** Linux x86-64, GCC 13, single-core (CI benchmark)  
**Флаги компиляции:** `-O3 -march=native -fopenmp`

---

## HPC Gate Results

| Gate | Метрика | Результат | Цель | Статус |
|------|---------|-----------|------|--------|
| 1 | HandEvaluator (7-card) | **6.8M ops/sec** | > 1M (65M с SVML) | ✅ PASS |
| 2 | BeliefTracker update | **19–39 µs** | < 45 µs | ✅ PASS |
| 3 | PointwiseOptimizer | **0.05–0.06 µs** | < 0.8 µs | ✅ PASS |
| 4 | HandSimulator throughput | **7.2K hands/s** | > 7K | ✅ PASS |
| 5 | NaturalGradient step | **< 0.01 ms** | < 8 ms | ✅ PASS |
| 6 | BAEM inference fraction | **22%** (incl. init) | < 30% | ✅ PASS |

### Производительность HandEvaluator

Прогресс по итерациям разработки:

| Реализация | Алгоритм | Throughput |
|------------|----------|-----------|
| v1 (Weeks 1-2) | Additive hash, C(7,5) enum | 0.5M ops/s |
| v2 (collisions fixed) | Polynomial LUT, C(7,5) enum | 1.5M ops/s |
| v3 (sorting network) | Polynomial LUT + CAS sort | 1.6M ops/s |
| **v4 (Weeks 9)** | **Histogram O(7), no enumeration** | **6.8M ops/s** |
| Production target | 2+2 LUT (32MB) + AVX-512 SVML | ~65M ops/s |

### Путь к 65M ops/sec

Для достижения production цели требуется:
1. **2+2 Evaluator** (`hand_evaluator_2plus2.hpp`) — 32MB pre-computed table, 7 lookups, ~15ns/eval
2. **AVX-512 SVML** — vectorised `log()` в EntropyCalculator, ~3µs для 1326 элементов
3. **Hugepage buffer** для LUT (устраняет TLB miss: 512×4KB → 1×2MB hugepage)

## Mathematical Validation

| Тест | Параметры | Результат |
|------|-----------|-----------|
| Тест 1: λ̂ сходимость | T=30K, λ_gen=0.35 | Стабилизация за 1K раздач ✓ |
| Тест 2: Concept Drift | λ: 0.15→0.75 | Детекция за 73 раздачи (лимит 400) ✓ |
| Тест 3: Dynamic Regret | T=10K, drift=10 | R_T/T = 0.09 < 0.5 ✓ |

## GTO Engine Convergence

| Игра | Итерации | Info Sets | Exploitability |
|------|----------|-----------|----------------|
| Kuhn Poker | 1000 | 12 | ~0.1 |
| Leduc Hold'em | 5000 | ~23 | ~0.01 |
| NLHE Preflop HU | 10000 | 845 | ~0.009 |

## Latency Breakdown (on_opponent_action)

```
BeliefTracker::update()      ~19 µs  (dominant: 1326 × likelihood computation)
ExploitabilityEstimator      ~0.5 µs (KL divergence, 5 actions)
KalmanPenaltyFilter::step()  ~0.1 µs
OnlineTrainer (every 16th)   ~2 ms   (MLP backward + Adam step)
```

## Memory Footprint

| Компонент | Память |
|-----------|--------|
| rank_lut_ (HandEvaluator) | 742 KB |
| flush_lut_ | 16 KB |
| BeliefTracker range | ~5 KB (1326 × float) |
| PokerHistoryTransformer | ~500 KB (64→64→5 MLP) |
| CFR InfoMap (NLHE Preflop) | ~200 KB (845 entries × 240 bytes) |
| **Итого** | **~1.5 MB** |
