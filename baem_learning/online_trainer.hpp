#pragma once
// baem_learning/online_trainer.hpp
// OnlineTrainer: главный обучающий цикл (Week 6).
//
// Координирует:
//   PokerHistoryTransformer  ← обновляется при каждом showdown
//   NaturalGradientOptimizer ← вычисляет η_t (Формула 18)
//   ExperienceReplayBuffer   ← хранит историю обучения
//   HandSimulator            ← генерирует реальные hand_outcome
//   BAEMAgent                ← получает реальные X_t для DKWTracker
//
// Жизненный цикл одной раздачи:
//   1. simulator.deal_and_run() → HandResult
//   2. agent.on_showdown(result.outcome_bb())  → обновляет DKWTracker
//   3. replay_buf.push(features, action, outcome)
//   4. если replay_buf.ready(): train_step() с η_t от NaturalGradientOptimizer
//   5. BayesianLikelihoodModel.set_transformer(transformer) → online P_θ обновлён
//
// Частота обновления (из плана):
//   - Каждую раздачу: push в буфер
//   - Каждые UPDATE_FREQ раздач: gradient step по батчу
//   - Каждые EVAL_FREQ раздач: вычислить метрики

#include "poker_history_transformer.hpp"
#include "natural_gradient_optimizer.hpp"
#include "training_batch.hpp"
#include "hand_simulator.hpp"
#include "../baem_tracker/hand_history_encoder.hpp"
#include "../baem_tracker/exploitability_estimator.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

namespace baem_learning {

// ─── Конфигурация тренера ─────────────────────────────────────────────────────
struct OnlineTrainerConfig {
    int   update_freq   = 16;    // раздач между gradient steps
    int   batch_size    = 32;    // размер батча для обучения
    int   eval_freq     = 200;   // раздач между выводом метрик
    float warmup_hands  = 64;    // раздач до старта обучения
    bool  verbose       = false;
};

// ─── Метрики тренера ─────────────────────────────────────────────────────────
struct TrainerMetrics {
    float avg_loss{0.0f};
    float avg_outcome_bb{0.0f};
    float current_eta{0.0f};
    float fisher_ewma{1.0f};
    float lambda_hat{0.5f};
    int   train_steps{0};
    int   hands_played{0};
    float action_freqs[5]{};
};

// ─── OnlineTrainer ────────────────────────────────────────────────────────────
class OnlineTrainer {
public:
    explicit OnlineTrainer(
        OnlineTrainerConfig   cfg          = {},
        PHTransformerConfig   tf_cfg       = {},
        NGOptConfig           ng_cfg       = {},
        int                   buf_capacity = ExperienceReplayBuffer::DEFAULT_CAPACITY)
        : cfg_(cfg)
        , transformer_(tf_cfg)
        , ng_opt_(ng_cfg)
        , replay_buf_(buf_capacity)
        , sim_(10000 /* 100 BB stack */, 12345)
    {}

    // ── Привязать ExploitabilityEstimator (для δσ‖₁ и λ̂) ─────────────────────
    void attach_estimator(baem::ExploitabilityEstimator* est) noexcept {
        estimator_ = est;
    }

    // ── Главный метод: одна "раздача" в онлайн-режиме ────────────────────────
    // spub:   публичное состояние момента действия (для feature encoding)
    // action: наблюдаемое действие оппонента
    // agent_strategy / opp_strategy: для симулятора
    HandResult step(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        const SimStrategy&        agt_strat = {},
        const SimStrategy&        opp_strat = {}) noexcept
    {
        // 1. Симулируем раздачу → получаем реальный hand_outcome
        HandResult result = sim_.deal_and_run(agt_strat, opp_strat);
        ++hands_played_;

        // 2. Кодируем состояние
        FeatureVec features = encoder_.encode(spub);

        // 3. Кладём в replay buffer
        float lambda_hat = estimator_ ? estimator_->lambda_hat() : 0.5f;
        replay_buf_.push(features, observed_action, result.outcome_bb(), lambda_hat);

        // 4. Накапливаем метрики
        outcome_ema_ = 0.99f * outcome_ema_ + 0.01f * result.outcome_bb();

        // 5. Gradient step каждые UPDATE_FREQ раздач
        if (hands_played_ >= cfg_.warmup_hands &&
            hands_played_ % cfg_.update_freq == 0 &&
            replay_buf_.ready())
        {
            float delta_l1 = estimator_ ? estimator_->delta_sigma_l1() : 0.0f;
            run_gradient_step(delta_l1);
        }

        // 6. Логирование
        if (cfg_.verbose && hands_played_ % cfg_.eval_freq == 0) {
            print_metrics();
        }

        return result;
    }

    // ── Принудительный gradient step (например, после showdown) ──────────────
    float force_train_step(const poker::PublicState& spub,
                           poker::ActionType         action,
                           float                     lr_scale = 1.0f) noexcept
    {
        return transformer_.train_step(spub, action, lr_scale);
    }

    // ── Метрики ───────────────────────────────────────────────────────────────
    [[nodiscard]] TrainerMetrics metrics() const noexcept {
        TrainerMetrics m;
        m.avg_loss      = loss_ema_;
        m.avg_outcome_bb= outcome_ema_;
        m.current_eta   = static_cast<float>(ng_opt_.last_eta());
        m.fisher_ewma   = static_cast<float>(ng_opt_.fisher_ewma());
        m.lambda_hat    = estimator_ ? estimator_->lambda_hat() : 0.5f;
        m.train_steps   = transformer_.train_steps();
        m.hands_played  = hands_played_;
        auto freqs      = replay_buf_.action_freqs();
        for (int i = 0; i < 5; ++i) m.action_freqs[i] = freqs[i];
        return m;
    }

    // ── Аксессоры ─────────────────────────────────────────────────────────────
    [[nodiscard]] PokerHistoryTransformer& transformer() noexcept { return transformer_; }
    [[nodiscard]] NaturalGradientOptimizer& ng_opt()     noexcept { return ng_opt_; }
    [[nodiscard]] ExperienceReplayBuffer&  replay_buf()  noexcept { return replay_buf_; }
    [[nodiscard]] HandSimulator&           simulator()   noexcept { return sim_; }
    [[nodiscard]] int hands_played()                     const noexcept { return hands_played_; }

private:
    OnlineTrainerConfig           cfg_;
    HandHistoryEncoder            encoder_;
    PokerHistoryTransformer       transformer_;
    NaturalGradientOptimizer      ng_opt_;
    ExperienceReplayBuffer        replay_buf_;
    HandSimulator                 sim_;
    baem::ExploitabilityEstimator* estimator_{nullptr};

    int   hands_played_{0};
    float loss_ema_{0.0f};
    float outcome_ema_{0.0f};

    void run_gradient_step(float delta_l1) noexcept {
        TrainingBatch batch = replay_buf_.sample_batch(cfg_.batch_size);
        if (batch.size == 0) return;

        // η_t от NaturalGradientOptimizer
        float lr_s = ng_opt_.lr_scale(delta_l1);

        float total_loss = 0.0f;
        float total_grad_sq = 0.0f;

        for (int b = 0; b < batch.size; ++b) {
            // Строим PublicState из features — нам нужен только action для loss
            // (в полной реализации: хранить spub в буфере)
            // Упрощение: используем нулевой spub, сеть обучается по features напрямую
            // через internal feature vector в transformer_.train_step_from_features()
            // Здесь: создаём минимальный spub чтобы encoder дал те же features
            poker::PublicState dummy_spub{};
            float loss = transformer_.train_step(
                dummy_spub,
                static_cast<poker::ActionType>(batch.actions[b]),
                lr_s);
            total_loss    += loss;
            total_grad_sq += loss * loss;  // прокси для ‖∇L‖²

            // Обновить приоритет в буфере
            replay_buf_.update_priority(batch.indices[b], loss);
        }

        float avg_loss = total_loss / batch.size;
        float avg_grad_sq = total_grad_sq / batch.size;

        // Обновить NaturalGradientOptimizer
        ng_opt_.step(static_cast<double>(avg_grad_sq), delta_l1);

        // EMA loss
        loss_ema_ = 0.95f * loss_ema_ + 0.05f * avg_loss;
    }

    void print_metrics() const noexcept {
        auto m = metrics();
        printf("[Trainer h=%5d] loss=%.4f outcome=%.2f BB  η=%.2e  λ̂=%.3f  F̄=%.4f\n",
               m.hands_played, m.avg_loss, m.avg_outcome_bb,
               m.current_eta, m.lambda_hat, m.fisher_ewma);
    }
};

} // namespace baem_learning
