#pragma once
// baem_learning/natural_gradient_optimizer.hpp  (Week 6 — полная реализация)
// NaturalGradientOptimizer: Formula (18) из BAEM v3 paper.
//
// η_t = η₀ · (I_F(θ_t) + λI)⁻¹ · (1 + ρ‖Δσ̂_opp‖₁)⁻¹
//
// Реализация:
//   1. Fisher Information Matrix аппроксимирован через EWMA градиентов²
//      (диагональное приближение — Gauss-Newton вариант Adam)
//   2. Tikhonov regularization λI добавляется к диагонали
//   3. Concept Drift term (1+ρ‖Δσ̂_opp‖₁)⁻¹ масштабирует η_t вниз при дрейфе
//
// Связь с unified_ml:
//   В HAVE_UNIFIED_ML режиме: используем core::Adam как inner optimizer,
//   накапливаем Fisher через grad² после каждого backward().
//   В standalone режиме: Fisher из AdamStateMLP.v (second moment = F_diag approx).
//
// Теоретическое свойство:
//   При σ_λ=0 (стационарный оппонент): η_t → η₀/F̄ → оптимальный шаг Ньютона.
//   При дрейфе: η_t ↓ автоматически (Формула 18).

#include "../baem_tracker/exploitability_estimator.hpp"
#include "poker_history_transformer.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

namespace baem_learning {

struct NGOptConfig {
    double eta0          = 3e-4;   // базовый learning rate
    double lambda_tik    = 1e-4;   // Tikhonov λ (регуляризация Fisher)
    double rho           = 10.0;   // чувствительность к Concept Drift
    double fisher_decay  = 0.999;  // EWMA для Fisher diagonal
    double min_eta       = 1e-6;   // нижний порог шага
    double max_eta       = 1e-2;   // верхний порог шага
    int    warmup_steps  = 100;    // шаги без Fischer-коррекции
};

class NaturalGradientOptimizer {
public:
    explicit NaturalGradientOptimizer(NGOptConfig cfg = {}) noexcept
        : cfg_(cfg) {}

    // ── Вычислить эффективный learning rate η_t ─────────────────────────────
    // fisher_diag_approx: F̄ = E[g²] (scalar approximation из PHTransformer)
    // delta_sigma_l1:    ‖Δσ̂_opp‖₁ из ExploitabilityEstimator
    [[nodiscard]] double compute_eta(
        double fisher_diag_approx,
        float  delta_sigma_l1,
        int    step = -1) const noexcept
    {
        // Warmup: без Fisher-коррекции
        int s = (step >= 0) ? step : step_;
        if (s < cfg_.warmup_steps)
            return cfg_.eta0;

        // η_t = η₀ / (F̄ + λ) / (1 + ρ·‖Δσ‖₁)
        double F_reg = fisher_diag_approx + cfg_.lambda_tik;
        double drift_term = 1.0 + cfg_.rho * static_cast<double>(delta_sigma_l1);
        double eta = cfg_.eta0 / (F_reg * drift_term);

        return std::clamp(eta, cfg_.min_eta, cfg_.max_eta);
    }

    // ── Шаг: обновить Fisher EMA и вернуть η_t ────────────────────────────────
    // Вызывается после каждого backward() в OnlineTrainer.
    // grad_sq: скалярная норма ‖∇L‖² текущего шага
    [[nodiscard]] double step(
        double grad_sq_norm,
        float  delta_sigma_l1) noexcept
    {
        ++step_;
        // EWMA обновление Fisher diagonal approximation
        fisher_ewma_ = cfg_.fisher_decay * fisher_ewma_
                     + (1.0 - cfg_.fisher_decay) * grad_sq_norm;

        double eta = compute_eta(fisher_ewma_, delta_sigma_l1, step_);
        last_eta_  = eta;
        return eta;
    }

    // ── Применить η_t к PokerHistoryTransformer (через lr_scale) ────────────
    // Возвращает lr_scale = η_t / η₀ для передачи в train_step()
    [[nodiscard]] float lr_scale(float delta_sigma_l1) noexcept {
        double eta = compute_eta(fisher_ewma_, delta_sigma_l1, step_);
        return static_cast<float>(eta / cfg_.eta0);
    }

    // ── Тест: дрейф уменьшает η ──────────────────────────────────────────────
    [[nodiscard]] bool drift_reduces_lr(float delta_before, float delta_after) const noexcept {
        return compute_eta(fisher_ewma_, delta_after) <
               compute_eta(fisher_ewma_, delta_before);
    }

    // ── Диагностика ───────────────────────────────────────────────────────────
    [[nodiscard]] double fisher_ewma()  const noexcept { return fisher_ewma_; }
    [[nodiscard]] double last_eta()     const noexcept { return last_eta_; }
    [[nodiscard]] int    num_steps()    const noexcept { return step_; }
    [[nodiscard]] const NGOptConfig& config() const noexcept { return cfg_; }

    void reset() noexcept {
        fisher_ewma_ = 1.0;
        last_eta_    = cfg_.eta0;
        step_        = 0;
    }

private:
    NGOptConfig cfg_;
    double      fisher_ewma_{1.0};   // скользящее F̄
    double      last_eta_{0.0};
    int         step_{0};
};

} // namespace baem_learning
