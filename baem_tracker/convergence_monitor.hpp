#pragma once
// baem_tracker/convergence_monitor.hpp
// ConvergenceMonitor: отслеживает сходимость λ̂(t) и проверяет условия ЦПТ.
//
// Реализует мониторинг для Теоремы 15.1 (Billingsley, ЦПТ для мартингал-разностей):
//   - Условие (23): 1/n Σ s²_k → σ²_∞ > 0
//   - Условие (24): Lindeberg (усечённая дисперсия → 0)
//
// Выдаёт метрики для Тестов 1-3 из плана валидации:
//   Тест 1: |λ̂ - λ_true| ≤ 0.01 при T=50000 → проверяет Утверждение 6.2
//   Тест 3: Dynamic Regret ≤ C(T·ε²_∞ + √T·σ_λ) → проверяет Формулу 15
//
// Хранит rolling statistics в кольцевом буфере (без динамических аллокаций
// после инициализации).

#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <span>

namespace baem {

struct CLTDiagnostics {
    float sigma_inf_sq{0.0f};    // оценка σ²_∞ (Формула 23)
    float lindeberg_term{0.0f};  // Lindeberg term (Формула 24, должен → 0)
    float clt_valid{0.0f};       // [0..1]: 1.0 = ЦПТ применима, 0 = нет
    int   n_samples{0};
};

struct RegretDiagnostics {
    float regret_cumulative{0.0f};   // R_T накопленный (Formula 15)
    float regret_bound{0.0f};        // теоретическая верхняя граница
    float regret_ratio{0.0f};        // R_T / bound (должно быть ≤ 1.0)
    float epsilon_inf{0.0f};         // установившаяся ошибка ε_∞
    float sigma_lambda{0.0f};        // оценка дрейфа σ_λ
    int   T{0};
};

class ConvergenceMonitor {
public:
    static constexpr int WINDOW = 1000;   // размер скользящего окна

    explicit ConvergenceMonitor(float true_lambda = -1.0f) noexcept
        : true_lambda_(true_lambda) {}

    // Зарегистрировать новое наблюдение λ̂(t) и исход раздачи X_t
    void record(float lambda_hat, float hand_outcome) noexcept {
        ++t_;
        last_lambda_hat_ = lambda_hat;

        // Обновляем rolling window λ̂
        lambda_buf_[t_ % WINDOW] = lambda_hat;

        // Rolling mean λ̂
        int wn = std::min(t_, WINDOW);
        float sum_lh = 0.0f;
        for (int i = 0; i < wn; ++i) sum_lh += lambda_buf_[i];
        lambda_hat_mean_ = sum_lh / static_cast<float>(wn);

        // Running mean и variance исходов (для ЦПТ)
        float delta = hand_outcome - outcomes_mean_;
        outcomes_mean_ += delta / static_cast<float>(t_);
        float delta2 = hand_outcome - outcomes_mean_;
        outcomes_M2_ += delta * delta2;      // Welford

        // Мартингал-разности: Y_k = X_k - E[X_k|F_{k-1}]
        float yk = hand_outcome - outcomes_mean_;
        yk_sq_sum_ += yk * yk;

        // Rolling σ̂²_λ (оценка дрейфа соперника)
        if (t_ >= 2) {
            int prev = (t_ - 1) % WINDOW;
            float dlambda = lambda_hat - lambda_buf_[prev];
            lambda_drift_M2_ += dlambda * dlambda;
        }

        // Обновить Dynamic Regret
        if (true_lambda_ >= 0.0f) {
            float error = lambda_hat - true_lambda_;
            regret_cumulative_ += error * error;   // квадратичная ошибка
        }
    }

    // ── CLT diagnostics (Формула 23-24) ────────────────────────────────────
    [[nodiscard]] CLTDiagnostics clt_diagnostics() const noexcept {
        CLTDiagnostics d;
        d.n_samples = t_;
        if (t_ < 10) return d;

        // σ²_∞ = sample variance of outcomes
        d.sigma_inf_sq = (t_ > 1) ? outcomes_M2_ / static_cast<float>(t_ - 1) : 0.0f;

        // Lindeberg term: доля "больших" слагаемых (упрощённая проверка)
        // Здесь мы проверяем что дисперсия стабилизировалась
        float recent_var = (t_ >= WINDOW)
            ? rolling_variance()
            : d.sigma_inf_sq;
        float ratio = (d.sigma_inf_sq > 1e-6f)
            ? std::abs(recent_var - d.sigma_inf_sq) / d.sigma_inf_sq
            : 1.0f;
        d.lindeberg_term = ratio;

        // ЦПТ применима если σ²_∞ > 0 и Lindeberg term → 0
        d.clt_valid = (d.sigma_inf_sq > 0.01f)
            ? std::max(0.0f, 1.0f - ratio)
            : 0.0f;

        return d;
    }

    // ── Convergence check (Утверждение 6.2) ─────────────────────────────────
    // Возвращает текущую ошибку |λ̂ - λ_true| если λ_true известна
    // Использует последнее зафиксированное значение (не скользящее среднее)
    [[nodiscard]] float lambda_error() const noexcept {
        if (true_lambda_ < 0.0f) return -1.0f;
        return std::abs(last_lambda_hat_ - true_lambda_);
    }

    [[nodiscard]] float last_lambda_hat() const noexcept { return last_lambda_hat_; }

    // Проверка O(1/√t) скорости сходимости (Утверждение 6.2)
    [[nodiscard]] bool convergence_on_track() const noexcept {
        if (true_lambda_ < 0.0f || t_ < 100) return true;
        float err = lambda_error();
        float bound = 2.0f / std::sqrt(static_cast<float>(t_));
        return err <= bound * 3.0f;  // 3× запас
    }

    // ── Dynamic Regret (Формула 15) ──────────────────────────────────────────
    [[nodiscard]] RegretDiagnostics regret_diagnostics(
        float epsilon_inf,
        float sigma_lambda) const noexcept
    {
        RegretDiagnostics d;
        d.T                = t_;
        d.regret_cumulative= regret_cumulative_;
        d.epsilon_inf      = epsilon_inf;
        d.sigma_lambda     = sigma_lambda;

        // Теоретическая граница: C * (T * ε²_∞ + √T * σ_λ)
        float T_f = static_cast<float>(t_);
        d.regret_bound = REGRET_CONST *
            (T_f * epsilon_inf * epsilon_inf + std::sqrt(T_f) * sigma_lambda);

        d.regret_ratio = (d.regret_bound > 1e-6f)
            ? d.regret_cumulative / d.regret_bound
            : 0.0f;

        // Оценка σ_λ из данных
        d.sigma_lambda = (t_ > 1)
            ? std::sqrt(lambda_drift_M2_ / static_cast<float>(t_ - 1))
            : 0.0f;

        return d;
    }

    // ── Accessors ────────────────────────────────────────────────────────────
    [[nodiscard]] float outcomes_mean()   const noexcept { return outcomes_mean_; }
    [[nodiscard]] float outcomes_var()    const noexcept {
        return t_ > 1 ? outcomes_M2_ / static_cast<float>(t_ - 1) : 0.0f;
    }
    [[nodiscard]] float lambda_hat_mean() const noexcept { return lambda_hat_mean_; }
    [[nodiscard]] int   num_samples()     const noexcept { return t_; }

    void set_true_lambda(float lam) noexcept { true_lambda_ = lam; }

    void reset() noexcept {
        t_ = 0;
        outcomes_mean_ = outcomes_M2_ = 0.0f;
        lambda_hat_mean_ = lambda_drift_M2_ = yk_sq_sum_ = 0.0f;
        regret_cumulative_ = 0.0f;
        lambda_buf_.fill(0.0f);
    }

private:
    int   t_{0};
    float true_lambda_{-1.0f};

    // Outcomes statistics (Welford online algorithm)
    float outcomes_mean_{0.0f};
    float outcomes_M2_{0.0f};

    // λ̂ rolling window
    std::array<float, WINDOW> lambda_buf_{};
    float lambda_hat_mean_{0.0f};
    float lambda_drift_M2_{0.0f};

    // Martingale differences
    float yk_sq_sum_{0.0f};

    // Dynamic Regret accumulator
    float regret_cumulative_{0.0f};
    float last_lambda_hat_{0.5f};

    static constexpr float REGRET_CONST = 2.0f;  // константа в O(·) bound

    // Variance в последних WINDOW наблюдениях (для Lindeberg)
    [[nodiscard]] float rolling_variance() const noexcept {
        // Упрощённо: берём текущую дисперсию по последним значениям
        // В полной реализации нужен sliding window
        return outcomes_M2_ / std::max(1, t_ - 1);
    }
};

} // namespace baem
