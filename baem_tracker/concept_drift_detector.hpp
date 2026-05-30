#pragma once
// baem_tracker/concept_drift_detector.hpp
// ConceptDriftDetector: обнаруживает резкую смену стиля игры соперника.
//
// Математический инструмент: CUSUM (Cumulative Sum Control Chart)
//   — оптимальный последовательный критерий обнаружения изменения
//     параметра распределения (тест Лоурена, 1954).
//
// Параллельно ведёт два показателя:
//   1. Δσ_opp L1 drift (из OppStrategyTracker) — быстрый детектор
//   2. CUSUM над λ̂(t) — долгосрочный детектор тренда
//
// При обнаружении дрейфа выдаёт:
//   - DriftEvent с уровнем серьёзности [mild/strong/critical]
//   - Рекомендацию для KalmanPenaltyFilter (inflation factor)
//   - Рекомендацию для NaturalGradientOptimizer (шаг обучения)
//
// Тест 2 из плана: выход на новую траекторию ≤ 400 раздач после смены стиля.

#include <cmath>
#include <cstdint>
#include <array>
#include <algorithm>
#include <optional>

namespace baem {

// ─── Уровень дрейфа ──────────────────────────────────────────────────────────
enum class DriftLevel : uint8_t {
    None     = 0,
    Mild     = 1,   // Δσ > threshold_mild
    Strong   = 2,   // Δσ > threshold_strong  ИЛИ  CUSUM > h_cusum
    Critical = 3,   // оба критерия сработали одновременно
};

// ─── Событие дрейфа ──────────────────────────────────────────────────────────
struct DriftEvent {
    DriftLevel level{DriftLevel::None};
    float      delta_l1{0.0f};          // L1 норма изменения стратегии
    float      cusum_stat{0.0f};        // текущая CUSUM статистика
    float      kalman_inflation{1.0f};  // рекомендуемое раздутие ε² для Kalman
    float      lr_scale{1.0f};          // рекомендуемый масштаб шага обучения
    int        hand_number{0};          // номер раздачи, когда произошёл дрейф
    bool       detected{false};

    operator bool() const noexcept { return detected; }
};

// ─── ConceptDriftDetector ─────────────────────────────────────────────────────
struct ConceptDriftDetectorConfig {
        // L1-пороги быстрого детектора
        float threshold_mild   = 0.08f;
        float threshold_strong = 0.18f;
        float threshold_critical = 0.30f;

        // CUSUM параметры
        float cusum_mu0 = 0.5f;    // ожидаемое λ̂ при стационарном сопернике
        float cusum_k   = 0.10f;   // допустимое отклонение (allowable slack)
        float cusum_h   = 4.0f;    // порог обнаружения (decision interval)

        // Kalman inflation при дрейфе
        float kalman_inflate_mild     =  2.0f;
        float kalman_inflate_strong   =  8.0f;
        float kalman_inflate_critical = 25.0f;

        // LR scale при дрейфе (Формула 18: η_t ↑ при дрейфе)
        float lr_scale_mild     = 2.0f;
        float lr_scale_strong   = 5.0f;
        float lr_scale_critical = 10.0f;

        // Cooldown: не срабатывать чаще чем раз в N раздач
        int cooldown_hands = 50;
};

class ConceptDriftDetector {
public:
    using Config = ConceptDriftDetectorConfig;

    explicit ConceptDriftDetector(Config cfg = ConceptDriftDetectorConfig{}) noexcept
        : cfg_(cfg) {}

    // Обновить состояние детектора.
    // delta_l1:   ||Δσ̂_opp||₁ из OppStrategyTracker
    // lambda_hat: текущий λ̂(t) из ExploitabilityEstimator
    // hand_num:   номер раздачи (для cooldown)
    [[nodiscard]] DriftEvent update(
        float delta_l1,
        float lambda_hat,
        int   hand_num) noexcept
    {
        // ── L1 быстрый детектор ────────────────────────────────────────────
        DriftLevel l1_level = DriftLevel::None;
        if      (delta_l1 >= cfg_.threshold_critical) l1_level = DriftLevel::Critical;
        else if (delta_l1 >= cfg_.threshold_strong)   l1_level = DriftLevel::Strong;
        else if (delta_l1 >= cfg_.threshold_mild)     l1_level = DriftLevel::Mild;

        // ── CUSUM детектор изменения λ̂ ────────────────────────────────────
        // Верхний CUSUM: обнаружение роста λ̂ (оппонент стал ближе к GTO)
        float z    = lambda_hat - cfg_.cusum_mu0;
        cusum_up_  = std::max(0.0f, cusum_up_  + z - cfg_.cusum_k);
        cusum_dn_  = std::max(0.0f, cusum_dn_  - z - cfg_.cusum_k);

        bool cusum_fired = (cusum_up_ > cfg_.cusum_h || cusum_dn_ > cfg_.cusum_h);
        float cusum_stat = std::max(cusum_up_, cusum_dn_);

        // Экспоненциальная скользящая для сглаживания delta_l1
        ema_delta_l1_ = 0.7f * ema_delta_l1_ + 0.3f * delta_l1;

        // ── Синтез уровня дрейфа ──────────────────────────────────────────
        DriftLevel level = l1_level;
        if (cusum_fired && level == DriftLevel::None) level = DriftLevel::Mild;
        if (cusum_fired && level == DriftLevel::Mild)   level = DriftLevel::Strong;
        if (cusum_fired && level >= DriftLevel::Strong)  level = DriftLevel::Critical;

        // ── Cooldown ──────────────────────────────────────────────────────
        if (level > DriftLevel::None) {
            if (hand_num - last_event_hand_ < cfg_.cooldown_hands) {
                // Обновляем статистику, но не генерируем новое событие
                last_delta_l1_ = delta_l1;
                return {};
            }
        }

        // ── Сброс CUSUM после срабатывания ────────────────────────────────
        bool detected = (level > DriftLevel::None);
        if (detected) {
            cusum_up_ = 0.0f;
            cusum_dn_ = 0.0f;
            last_event_hand_ = hand_num;
            ++total_events_;
        }

        last_delta_l1_ = delta_l1;
        last_level_ = level;

        if (!detected) return {};

        // ── Формируем DriftEvent ──────────────────────────────────────────
        DriftEvent ev;
        ev.detected     = true;
        ev.level        = level;
        ev.delta_l1     = delta_l1;
        ev.cusum_stat   = cusum_stat;
        ev.hand_number  = hand_num;

        switch (level) {
            case DriftLevel::Mild:
                ev.kalman_inflation = cfg_.kalman_inflate_mild;
                ev.lr_scale         = cfg_.lr_scale_mild;
                break;
            case DriftLevel::Strong:
                ev.kalman_inflation = cfg_.kalman_inflate_strong;
                ev.lr_scale         = cfg_.lr_scale_strong;
                break;
            case DriftLevel::Critical:
                ev.kalman_inflation = cfg_.kalman_inflate_critical;
                ev.lr_scale         = cfg_.lr_scale_critical;
                break;
            default: break;
        }

        return ev;
    }

    // Принудительный сброс (смена соперника / новая сессия)
    void reset() noexcept {
        cusum_up_ = cusum_dn_ = 0.0f;
        ema_delta_l1_ = 0.0f;
        last_event_hand_ = -1000;
        last_level_ = DriftLevel::None;
        total_events_ = 0;
    }

    // ── Доступ к состоянию ────────────────────────────────────────────────
    [[nodiscard]] float      cusum_up()    const noexcept { return cusum_up_; }
    [[nodiscard]] float      cusum_dn()    const noexcept { return cusum_dn_; }
    [[nodiscard]] float      ema_delta()   const noexcept { return ema_delta_l1_; }
    [[nodiscard]] DriftLevel last_level()  const noexcept { return last_level_; }
    [[nodiscard]] int        total_events()const noexcept { return total_events_; }
    [[nodiscard]] const Config& config()   const noexcept { return cfg_; }

private:
    Config cfg_;
    float  cusum_up_{0.0f};        // верхняя CUSUM
    float  cusum_dn_{0.0f};        // нижняя CUSUM
    float  ema_delta_l1_{0.0f};    // EMA сглаживание
    float  last_delta_l1_{0.0f};
    int    last_event_hand_{-1000};
    DriftLevel last_level_{DriftLevel::None};
    int    total_events_{0};
};

} // namespace baem
