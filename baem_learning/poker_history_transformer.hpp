#pragma once
// baem_learning/poker_history_transformer.hpp
// PokerHistoryTransformer: P_θ(a | h, board, s_pub)
//
// Архитектура (Week 5):
//   Input:  float[FEATURE_DIM=64] из HandHistoryEncoder → double[] → Tensor[1, SEQ_LEN, EMBED_DIM]
//   Model:  TransformerEncoder (unified_ml) — encode + linear head [EMBED_DIM → NUM_ACTIONS]
//   Output: softmax probabilities float[5] для каждого из 1326 комбо
//
// Два режима компиляции:
//   HAVE_UNIFIED_ML=1  → реальный TransformerEncoder из unified_ml
//   HAVE_UNIFIED_ML=0  → автономный MLP-fallback (2-слойная сеть на hpc_kernels)
//
// Интерфейс намеренно одинаков в обоих режимах — BAEMAgent не знает о наличии зависимости.
//
// Интеграция с BayesianLikelihoodModel (Week 5):
//   transformer.compute_likelihood(spub, action, range, out_lh[1326])
//   заменяет аналитическую аппроксимацию аналогичным вызовом.
//
// Обучение (Week 6 / OnlineTrainer):
//   transformer.train_step(feature_vec, observed_action)
//   → forward → cross_entropy_loss → backward → Adam.step()

#include "../baem_tracker/hand_history_encoder.hpp"
#include "../baem_tracker/range_conditioner.hpp"
#include "../poker_core/game_state.hpp"

#ifdef HAVE_UNIFIED_ML
#  include "models/transformer/transformer_block.hpp"
#  include "core/optimizers.hpp"
#  include "autograd/tensor.h"
#endif

#include <array>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <cassert>
#include <random>
#include <numeric>

namespace baem_learning {

// Импорт из baem namespace
using baem::FEATURE_DIM;
using baem::COMBO_COUNT;
using baem::FeatureVec;
using baem::HandHistoryEncoder;
using baem::RangeBuffer;

// ─── Конфигурация трансформера ──────────────────────────────────────────────
struct PHTransformerConfig {
    // Трансформер
    int   embed_dim   = 32;   // размерность эмбеддинга (кратно num_heads)
    int   ff_hidden   = 128;  // скрытый слой FF
    int   num_heads   = 4;    // головы внимания
    int   num_layers  = 2;    // число блоков
    int   seq_len     = 1;    // длина последовательности (1 = текущий state)

    // Входные/выходные размерности
    static constexpr int INPUT_DIM  = baem::FEATURE_DIM;  // 64
    static constexpr int OUTPUT_DIM = 5;             // действия

    // Обучение
    double lr         = 3e-4;
    double beta1      = 0.9;
    double beta2      = 0.999;
    double weight_decay = 1e-4;

    // Inference
    float  temperature = 1.0f; // для softmax вывода
};

// ─── Standalone MLP-fallback (без unified_ml) ─────────────────────────────────
// 2-слойная сеть: [INPUT_DIM → hidden → OUTPUT_DIM], веса + biases как std::vector<double>
// Обучение: SGD с численным градиентом (для тестирования без зависимостей)
struct MLPWeights {
    static constexpr int IN  = PHTransformerConfig::INPUT_DIM;   // 64
    static constexpr int HID = 64;
    static constexpr int OUT = PHTransformerConfig::OUTPUT_DIM;  // 5

    std::vector<double> W1, b1, W2, b2;  // [HID×IN], [HID], [OUT×HID], [OUT]

    MLPWeights() {
        W1.resize(HID * IN);
        b1.resize(HID, 0.0);
        W2.resize(OUT * HID);
        b2.resize(OUT, 0.0);
        // Xavier initialization
        std::mt19937_64 rng(42);
        double scale1 = std::sqrt(2.0 / IN);
        double scale2 = std::sqrt(2.0 / HID);
        std::normal_distribution<double> nd1(0.0, scale1);
        std::normal_distribution<double> nd2(0.0, scale2);
        for (auto& w : W1) w = nd1(rng);
        for (auto& w : W2) w = nd2(rng);
    }

    // Forward: x[IN] → logits[OUT]
    void forward(const double* x, double* logits) const noexcept {
        // Layer 1: h = ReLU(W1 @ x + b1)
        double h[HID];
        for (int i = 0; i < HID; ++i) {
            double z = b1[i];
            for (int j = 0; j < IN; ++j) z += W1[i * IN + j] * x[j];
            h[i] = z > 0.0 ? z : 0.0;  // ReLU
        }
        // Layer 2: logits = W2 @ h + b2
        for (int i = 0; i < OUT; ++i) {
            double z = b2[i];
            for (int j = 0; j < HID; ++j) z += W2[i * HID + j] * h[j];
            logits[i] = z;
        }
    }

    // Softmax + probabilities [OUT]
    void forward_probs(const double* x, float* probs) const noexcept {
        double logits[OUT];
        forward(x, logits);
        // Stable softmax
        double mx = *std::max_element(logits, logits + OUT);
        double sum = 0.0;
        double exp_logits[OUT];
        for (int i = 0; i < OUT; ++i) { exp_logits[i] = std::exp(logits[i] - mx); sum += exp_logits[i]; }
        for (int i = 0; i < OUT; ++i) probs[i] = static_cast<float>(exp_logits[i] / sum);
    }
};

// ─── Adam state для MLP-fallback ─────────────────────────────────────────────
struct AdamStateMLP {
    std::vector<double> mW1, vW1, mb1, vb1;
    std::vector<double> mW2, vW2, mb2, vb2;
    int t{0};
    double lr, beta1, beta2, eps, wd;

    AdamStateMLP(const MLPWeights& w, double lr_, double b1=0.9, double b2=0.999,
                 double eps_=1e-8, double wd_=1e-4)
        : mW1(w.W1.size(), 0), vW1(w.W1.size(), 0)
        , mb1(w.b1.size(), 0), vb1(w.b1.size(), 0)
        , mW2(w.W2.size(), 0), vW2(w.W2.size(), 0)
        , mb2(w.b2.size(), 0), vb2(w.b2.size(), 0)
        , lr(lr_), beta1(b1), beta2(b2), eps(eps_), wd(wd_) {}

    // Adam update for one parameter array
    static void adam_update(double* param, double* grad,
                            double* m, double* v,
                            int n, int t,
                            double lr, double b1, double b2,
                            double eps, double wd) {
        double bc1 = 1.0 - std::pow(b1, t);
        double bc2 = 1.0 - std::pow(b2, t);
        for (int i = 0; i < n; ++i) {
            if (wd > 0.0) param[i] *= (1.0 - lr * wd);
            m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
            v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
            double mhat = m[i] / bc1;
            double vhat = v[i] / bc2;
            param[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
};

// ─── PokerHistoryTransformer ─────────────────────────────────────────────────
class PokerHistoryTransformer {
public:
    static constexpr int INPUT_DIM  = PHTransformerConfig::INPUT_DIM;
    static constexpr int OUTPUT_DIM = PHTransformerConfig::OUTPUT_DIM;

    explicit PokerHistoryTransformer(PHTransformerConfig cfg = {}) noexcept
        : cfg_(cfg), encoder_(cfg), adam_state_(encoder_.weights_, cfg.lr)
    {
#ifdef HAVE_UNIFIED_ML
        init_unified_ml();
#endif
    }

    // ── Inference: s_pub → P(a | combo_k) для 1326 комбо ─────────────────────
    // Вычисляет feature vector для s_pub, прогоняет через сеть,
    // возвращает P_θ(observed_action | s_pub) как единый скаляр.
    // Для разных комбо: в standalone-режиме prob не зависит от combo карт
    // (только от публичного состояния) — это корректно: мы не знаем карты соперника.
    [[nodiscard]] float infer_action_prob(
        const poker::PublicState& spub,
        poker::ActionType         observed_action) const noexcept
    {
        FeatureVec fv = hist_encoder_.encode(spub);
        float probs[OUTPUT_DIM];
        compute_probs(fv.data(), probs);
        float p = probs[static_cast<int>(observed_action)];
        return std::max(p, 1e-4f);
    }

    // ── Inference: заполнить lh[COMBO_COUNT] для BeliefTracker ──────────────
    // В standalone-режиме: все комбо получают одинаковый P(a|s_pub)
    // (публичная информация одинакова для всех рук оппонента).
    // В unified_ml режиме: можно добавить combo-specific conditioning.
    void compute_likelihood(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        const RangeBuffer&        range_buf,
        float*                    out_lh) const noexcept
    {
        FeatureVec fv = hist_encoder_.encode(spub);
        float probs[OUTPUT_DIM];
        compute_probs(fv.data(), probs);
        float p_action = std::max(probs[static_cast<int>(observed_action)], 1e-4f);

        // Для каждого комбо: P(a|h_k, s_pub) = P(a|s_pub) (публичная инвариантность)
        // Будущее расширение (combo conditioning): добавить board_strength как доп. фичу
        for (int k = 0; k < COMBO_COUNT; ++k) {
            out_lh[k] = range_buf.active[k] ? p_action : 0.0f;
        }
    }

    // ── Train step: одно обновление после наблюдения действия ───────────────
    // Потеря: cross-entropy L = -log P_θ(observed_action | s_pub)
    // Returns: loss value (для мониторинга)
    float train_step(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        float                     lr_scale = 1.0f) noexcept
    {
#ifdef HAVE_UNIFIED_ML
        return train_step_unified(spub, observed_action, lr_scale);
#else
        return train_step_standalone(spub, observed_action, lr_scale);
#endif
    }

    // ── Параметры для NaturalGradientOptimizer ────────────────────────────────
    // В standalone-режиме: возвращаем размер параметрического пространства
    [[nodiscard]] int num_parameters() const noexcept {
        return static_cast<int>(
            encoder_.weights_.W1.size() + encoder_.weights_.b1.size() +
            encoder_.weights_.W2.size() + encoder_.weights_.b2.size());
    }

    // ── Fisher Information diagonal (скалярное приближение) ──────────────────
    // Используется в NaturalGradientOptimizer для масштабирования шага.
    // Аппроксимация: F_diag ≈ E[∇log P_θ · ∇log P_θ] ≈ running variance градиентов.
    [[nodiscard]] float fisher_diagonal_approx() const noexcept {
        return fisher_approx_;
    }

    // ── Сохранение/загрузка весов (бинарный формат) ──────────────────────────
    bool save_weights(const char* path) const noexcept;
    bool load_weights(const char* path) noexcept;

    // ── Аксессоры ─────────────────────────────────────────────────────────────
    [[nodiscard]] const PHTransformerConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] float last_loss()           const noexcept { return last_loss_; }
    [[nodiscard]] int   train_steps()         const noexcept { return train_steps_; }

private:
    PHTransformerConfig   cfg_;
    HandHistoryEncoder    hist_encoder_;

    // ── Standalone MLP (всегда доступен как fallback) ─────────────────────────
    struct StandaloneEncoder {
        MLPWeights weights_;
        explicit StandaloneEncoder(const PHTransformerConfig&) {}
    } encoder_;

    AdamStateMLP adam_state_;

    // ── Runtime state ─────────────────────────────────────────────────────────
    mutable float fisher_approx_{1.0f};
    float         last_loss_{0.0f};
    int           train_steps_{0};

    // ── Core inference (dispatch) ─────────────────────────────────────────────
    void compute_probs(const float* feat_f32, float* probs) const noexcept {
#ifdef HAVE_UNIFIED_ML
        compute_probs_unified(feat_f32, probs);
#else
        // Convert f32 → f64 for MLPWeights
        double feat_f64[64];  // INPUT_DIM=64
        for (int i = 0; i < INPUT_DIM; ++i) feat_f64[i] = static_cast<double>(feat_f32[i]);
        encoder_.weights_.forward_probs(feat_f64, probs);
#endif
    }

    // ── Standalone training ───────────────────────────────────────────────────
    float train_step_standalone(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        float                     lr_scale) noexcept
    {
        FeatureVec fv = hist_encoder_.encode(spub);
        double feat[64];  // INPUT_DIM=64
        for (int i = 0; i < INPUT_DIM; ++i) feat[i] = static_cast<double>(fv[i]);

        const int target = static_cast<int>(observed_action);
        MLPWeights& W = encoder_.weights_;

        // ── Forward pass ──────────────────────────────────────────────────────
        double h[MLPWeights::HID], z1[MLPWeights::HID];
        for (int i = 0; i < MLPWeights::HID; ++i) {
            double z = W.b1[i];
            for (int j = 0; j < INPUT_DIM; ++j) z += W.W1[i * INPUT_DIM + j] * feat[j];
            z1[i] = z;
            h[i]  = z > 0.0 ? z : 0.0;
        }
        double logits[OUTPUT_DIM];
        W.forward(feat, logits);  // reuse to get logits

        // Softmax + cross-entropy loss
        double mx = *std::max_element(logits, logits + OUTPUT_DIM);
        double exps[OUTPUT_DIM], sum = 0.0;
        for (int i = 0; i < OUTPUT_DIM; ++i) { exps[i] = std::exp(logits[i] - mx); sum += exps[i]; }
        double probs_d[OUTPUT_DIM];
        for (int i = 0; i < OUTPUT_DIM; ++i) probs_d[i] = exps[i] / sum;
        float loss = static_cast<float>(-std::log(std::max(probs_d[target], 1e-9)));

        // ── Backward pass (manual backprop) ───────────────────────────────────
        // dL/dlogits = probs - one_hot(target)
        double dlogits[OUTPUT_DIM];
        for (int i = 0; i < OUTPUT_DIM; ++i) dlogits[i] = probs_d[i];
        dlogits[target] -= 1.0;

        // dL/dW2[i,j] = dlogits[i] * h[j]; dL/db2[i] = dlogits[i]
        double dW2[OUTPUT_DIM * MLPWeights::HID]{};
        double db2[OUTPUT_DIM]{};
        double dh[MLPWeights::HID]{};
        for (int i = 0; i < OUTPUT_DIM; ++i) {
            db2[i] = dlogits[i];
            for (int j = 0; j < MLPWeights::HID; ++j) {
                dW2[i * MLPWeights::HID + j] = dlogits[i] * h[j];
                dh[j] += dlogits[i] * W.W2[i * MLPWeights::HID + j];
            }
        }

        // dL/dz1[i] = dh[i] * (z1[i]>0) (ReLU gradient)
        double dz1[MLPWeights::HID]{};
        for (int i = 0; i < MLPWeights::HID; ++i) dz1[i] = dh[i] * (z1[i] > 0.0 ? 1.0 : 0.0);

        // dL/dW1[i,j] = dz1[i] * feat[j]; dL/db1[i] = dz1[i]
        double dW1[MLPWeights::HID * 64]{};  // INPUT_DIM=64
        double db1[MLPWeights::HID]{};
        for (int i = 0; i < MLPWeights::HID; ++i) {
            db1[i] = dz1[i];
            for (int j = 0; j < INPUT_DIM; ++j)
                dW1[i * INPUT_DIM + j] = dz1[i] * feat[j];
        }

        // ── Adam update ───────────────────────────────────────────────────────
        double eff_lr = cfg_.lr * static_cast<double>(lr_scale);
        adam_state_.t++;
        AdamStateMLP::adam_update(W.W1.data(), dW1, adam_state_.mW1.data(), adam_state_.vW1.data(),
            MLPWeights::HID * INPUT_DIM, adam_state_.t, eff_lr, cfg_.beta1, cfg_.beta2, 1e-8, cfg_.weight_decay);
        AdamStateMLP::adam_update(W.b1.data(), db1, adam_state_.mb1.data(), adam_state_.vb1.data(),
            MLPWeights::HID, adam_state_.t, eff_lr, cfg_.beta1, cfg_.beta2, 1e-8, 0.0);
        AdamStateMLP::adam_update(W.W2.data(), dW2, adam_state_.mW2.data(), adam_state_.vW2.data(),
            OUTPUT_DIM * MLPWeights::HID, adam_state_.t, eff_lr, cfg_.beta1, cfg_.beta2, 1e-8, cfg_.weight_decay);
        AdamStateMLP::adam_update(W.b2.data(), db2, adam_state_.mb2.data(), adam_state_.vb2.data(),
            OUTPUT_DIM, adam_state_.t, eff_lr, cfg_.beta1, cfg_.beta2, 1e-8, 0.0);

        // ── Fisher approximation update (EMA of grad²) ────────────────────────
        double grad_sq = 0.0;
        for (auto g : dlogits) grad_sq += g * g;
        fisher_approx_ = 0.95f * fisher_approx_ + 0.05f * static_cast<float>(grad_sq);

        last_loss_   = loss;
        ++train_steps_;
        return loss;
    }

#ifdef HAVE_UNIFIED_ML
    // ── unified_ml TransformerEncoder backend ─────────────────────────────────
    // Создаём при первом вызове (lazy init) чтобы не блокировать standalone-путь.
    mutable std::unique_ptr<transformer::TransformerEncoder> tf_encoder_;
    mutable std::unique_ptr<core::Adam>                      tf_optimizer_;

    void init_unified_ml() {
        transformer::TransformerConfig tf_cfg;
        tf_cfg.embed_dim   = static_cast<std::size_t>(cfg_.embed_dim);
        tf_cfg.ff_hidden_dim = static_cast<std::size_t>(cfg_.ff_hidden);
        tf_cfg.num_heads   = static_cast<std::size_t>(cfg_.num_heads);
        tf_cfg.num_layers  = static_cast<std::size_t>(cfg_.num_layers);
        tf_cfg.num_classes = static_cast<std::size_t>(OUTPUT_DIM);
        tf_cfg.max_seq_len = 8;       // история последних 8 действий
        tf_cfg.causal      = false;   // энкодер, не авторегрессивный

        tf_encoder_ = std::make_unique<transformer::TransformerEncoder>(tf_cfg);
        tf_optimizer_ = std::make_unique<core::Adam>(
            tf_encoder_->parameters(),
            cfg_.lr, cfg_.beta1, cfg_.beta2, 1e-8, cfg_.weight_decay);
    }

    void compute_probs_unified(const float* feat_f32, float* probs) const noexcept {
        if (!tf_encoder_) { // fallback
            double feat_f64[64];  // INPUT_DIM=64
            for (int i = 0; i < INPUT_DIM; ++i) feat_f64[i] = feat_f32[i];
            encoder_.weights_.forward_probs(feat_f64, probs);
            return;
        }
        // INPUT_DIM(64) → embed_dim через linear projection (здесь: pad/truncate для простоты)
        // В production: добавить input_projection слой
        std::size_t E = static_cast<std::size_t>(cfg_.embed_dim);
        std::vector<double> input_data(E, 0.0);
        int copy_n = std::min(INPUT_DIM, static_cast<int>(E));
        for (int i = 0; i < copy_n; ++i) input_data[i] = static_cast<double>(feat_f32[i]);

        // Shape: [1, 1, embed_dim] — batch=1, seq_len=1
        autograd::Tensor x(input_data, {1, 1, E}, false);
        autograd::Tensor logits = tf_encoder_->classify(x);

        // Softmax
        const double* ldata = logits.data_ptr();
        int n = static_cast<int>(logits.numel());
        double mx = *std::max_element(ldata, ldata + n);
        double sum = 0.0;
        std::vector<double> exps(n);
        for (int i = 0; i < n; ++i) { exps[i] = std::exp(ldata[i] - mx); sum += exps[i]; }
        for (int i = 0; i < std::min(n, OUTPUT_DIM); ++i)
            probs[i] = static_cast<float>(exps[i] / sum);
    }

    float train_step_unified(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        float                     lr_scale) noexcept
    {
        if (!tf_encoder_) return train_step_standalone(spub, observed_action, lr_scale);

        FeatureVec fv = hist_encoder_.encode(spub);
        std::size_t E = static_cast<std::size_t>(cfg_.embed_dim);
        std::vector<double> input_data(E, 0.0);
        int copy_n = std::min(INPUT_DIM, static_cast<int>(E));
        for (int i = 0; i < copy_n; ++i) input_data[i] = static_cast<double>(fv[i]);

        autograd::Tensor x(input_data, {1, 1, E}, false);

        // Forward + cross-entropy via unified_ml autograd
        autograd::Tensor logits = tf_encoder_->classify(x);

        // Cross-entropy loss: L = -log softmax(logits)[target]
        autograd::Tensor log_probs = autograd::log(autograd::softmax(logits));
        int target = static_cast<int>(observed_action);
        const double* lp = log_probs.data_ptr();
        int n = static_cast<int>(log_probs.numel());
        target = std::min(target, n - 1);

        // scalar loss = -log_probs[target]
        std::vector<double> loss_data(1, -lp[target]);
        autograd::Tensor scalar_loss(loss_data, {}, true);

        tf_optimizer_->zero_grad();
        scalar_loss.backward();
        if (lr_scale != 1.0f)
            tf_optimizer_->set_learning_rate(cfg_.lr * static_cast<double>(lr_scale));
        tf_optimizer_->step();

        float loss = static_cast<float>(-lp[target]);
        last_loss_   = loss;
        ++train_steps_;
        return loss;
    }
#endif // HAVE_UNIFIED_ML
};

// ─── Save/Load реализация (inline) ───────────────────────────────────────────
inline bool PokerHistoryTransformer::save_weights(const char* path) const noexcept {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    auto write_vec = [&](const std::vector<double>& v) {
        std::size_t n = v.size();
        std::fwrite(&n, sizeof(n), 1, f);
        std::fwrite(v.data(), sizeof(double), n, f);
    };
    write_vec(encoder_.weights_.W1); write_vec(encoder_.weights_.b1);
    write_vec(encoder_.weights_.W2); write_vec(encoder_.weights_.b2);
    std::fclose(f);
    return true;
}

inline bool PokerHistoryTransformer::load_weights(const char* path) noexcept {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    auto read_vec = [&](std::vector<double>& v) {
        std::size_t n; std::fread(&n, sizeof(n), 1, f);
        v.resize(n); std::fread(v.data(), sizeof(double), n, f);
    };
    read_vec(encoder_.weights_.W1); read_vec(encoder_.weights_.b1);
    read_vec(encoder_.weights_.W2); read_vec(encoder_.weights_.b2);
    // Reinit Adam state
    adam_state_ = AdamStateMLP(encoder_.weights_, cfg_.lr);
    std::fclose(f);
    return true;
}

} // namespace baem_learning
