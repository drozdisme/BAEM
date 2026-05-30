#pragma once
// baem_tracker/bayesian_likelihood_model.hpp
// BayesianLikelihoodModel: вычисляет P_θ(a | h, board) для всех 1326 комбо.
//
// Это слой между BeliefTracker и будущим PokerHistoryTransformer.
// В Неделях 1-4: таблично-аналитический вариант (без нейросети).
// В Неделях 5-6: заменяется на вызов Transformer forward pass.
//
// Аналитическая модель:
//   P(a | h, board) ∝ GTO_weight(h, a) · exploit_adjustment(h, a, s_pub)
//   где exploit_adjustment зависит от λ̂ и equity(h, board).
//
// Интерфейс задокументирован для замены на нейросеть:
//   compute_likelihood(spub, observed_action, out_lh[COMBO_COUNT])
//
// Все вычисления без heap-аллокаций. out_lh — caller-allocated.

#include "range_conditioner.hpp"
#include "hand_history_encoder.hpp"
#include "../gto_engine/gto_oracle.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include <cmath>
#include <algorithm>
#include <array>
#include <span>

namespace baem {

class BayesianLikelihoodModel {
public:
    // λ_hat: текущая оценка GTO-отклонения оппонента [0..1]
    // oracle: GTO-оракул для базовых вероятностей
    explicit BayesianLikelihoodModel(
        const gto::IGTOracle* oracle,
        float lambda_hat = 0.5f) noexcept
        : oracle_(oracle), lambda_hat_(lambda_hat) {}

    void set_lambda_hat(float lh) noexcept { lambda_hat_ = std::clamp(lh, 0.0f, 1.0f); }
    void set_oracle(const gto::IGTOracle* o) noexcept { oracle_ = o; }

    // Главный метод: вычислить лайклихуд P_θ(a | h_k, board) для каждого из 1326 комбо.
    //
    // out_lh: массив float[COMBO_COUNT], заполняется вероятностями.
    //   Индексация соответствует combo_index(i,j).
    //   Неактивные комбо (range.active[k]==0) получают 0.0.
    //
    // Аналитическая модель (до интеграции нейросети):
    //   Для GTO-поведения (λ→1): P ≈ σ_GTO(a | bucket(h))
    //   Для эксплойт-поведения (λ→0): P смещается в сторону equity-обоснованных действий
    //
    // board_strength[k]: предвычисленная нормализованная сила руки комбо k (0..1)
    void compute_likelihood(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        const RangeConditioner&   range,
        std::span<const float>    board_strength,   // [COMBO_COUNT], нормализован 0..1
        float*                    out_lh) const noexcept
    {
        assert(static_cast<int>(board_strength.size()) == COMBO_COUNT);

        const auto& buf = range.buffer();
        int act_idx = static_cast<uint8_t>(observed_action);

        // Если задан трансформер — делегируем ему
        if (likelihood_fn_) {
            likelihood_fn_(likelihood_ctx_, spub, observed_action, buf, out_lh);
            return;
        }

        for (int k = 0; k < COMBO_COUNT; ++k) {
            if (!buf.active[k]) { out_lh[k] = 0.0f; continue; }

            float str = board_strength[k];  // нормализованная сила [0..1]

            // Приближение GTO: вероятность действия зависит от силы руки
            float p_gto = gto_action_prob(str, act_idx, spub);

            // Эксплойт-компонент: сильные руки чаще рейзят/колят, слабые — фолдят
            float p_exploit = exploit_action_prob(str, act_idx);

            // Смешивание: λ̂ * P_gto + (1-λ̂) * P_exploit
            float p_mixed = lambda_hat_ * p_gto + (1.0f - lambda_hat_) * p_exploit;

            // Минимум ε для избежания нулевых лайклихудов
            out_lh[k] = std::max(p_mixed, 1e-4f);
        }
    }

    // Предвычислить board_strength[k] для всех активных комбо.
    // Использует HandEvaluator для оценки относительной силы.
    // board: текущие карты борда
    // out_strength: float[COMBO_COUNT], caller-allocated
    void compute_board_strengths(
        poker::CardSet                board,
        const RangeConditioner&       range,
        const poker::HandEvaluator&   evaluator,
        float*                        out_strength) const noexcept
    {
        const auto& buf = range.buffer();

        // Два прохода: (1) считаем raw strength, (2) нормализуем на [0..1]
        uint16_t min_s = 65535, max_s = 0;
        static thread_local std::array<uint16_t, COMBO_COUNT> raw{};

        for (int k = 0; k < COMBO_COUNT; ++k) {
            if (!buf.active[k]) { raw[k] = 0; continue; }

            poker::Card c1{buf.c1_idx[k]};
            poker::Card c2{buf.c2_idx[k]};

            int board_cnt = board.size();
            if (board_cnt == 0) {
                // Preflop: используем preflop rank (ранк карт)
                int r1 = c1.rank(), r2 = c2.rank();
                bool suited = (c1.suit() == c2.suit());
                bool paired = (r1 == r2);
                if (r1 < r2) std::swap(r1, r2);
                // Упрощённая preflop сила [0..8191]
                uint16_t s = static_cast<uint16_t>(
                    (r1 * 13 + r2) * 2 + (suited ? 1 : 0) + (paired ? 200 : 0));
                raw[k] = s;
            } else if (board_cnt + 2 >= 5) {
                // Postflop (flop/turn/river): достаточно карт для оценки
                poker::CardSet full = board;
                full.add(c1); full.add(c2);
                raw[k] = evaluator.evaluate(full);
            } else {
                // Мало карт — используем rank-based heuristic
                int r1 = c1.rank(), r2 = c2.rank();
                if (r1 < r2) std::swap(r1, r2);
                raw[k] = static_cast<uint16_t>(r1 * 13 + r2 + board_cnt * 100);
            }

            if (raw[k] < min_s) min_s = raw[k];
            if (raw[k] > max_s) max_s = raw[k];
        }

        // Нормализация [0..1]
        float range_s = static_cast<float>(max_s - min_s);
        if (range_s < 1.0f) range_s = 1.0f;
        for (int k = 0; k < COMBO_COUNT; ++k) {
            if (!buf.active[k]) { out_strength[k] = 0.0f; continue; }
            out_strength[k] = static_cast<float>(raw[k] - min_s) / range_s;
        }
    }

    // ── Опциональный Transformer (Week 5-6) ─────────────────────────────────
    // Если задан — compute_likelihood использует P_θ из трансформера вместо аналитики.
    // Задаётся через set_transformer() из OnlineTrainer после инициализации.
    // Тип void* для избежания circular dependency; приводится через function ptr.
    using LikelihoodFn = void(*)(const void* ctx,
                                  const poker::PublicState& spub,
                                  poker::ActionType action,
                                  const RangeBuffer& buf,
                                  float* out_lh);

    void set_likelihood_fn(LikelihoodFn fn, const void* ctx) noexcept {
        likelihood_fn_ = fn;
        likelihood_ctx_ = ctx;
    }

private:
    const gto::IGTOracle* oracle_{nullptr};
    float lambda_hat_{0.5f};
    LikelihoodFn likelihood_fn_{nullptr};
    const void*  likelihood_ctx_{nullptr};

    // Аналитическая аппроксимация GTO-вероятности действия по силе руки
    static float gto_action_prob(float str, int act_idx,
                                  const poker::PublicState& spub) noexcept {
        bool facing_bet = (spub.pot.current_bet_bb100 > 0);
        // Упрощённая GTO: сильные руки = raise/call, слабые = fold/check
        switch (act_idx) {
            case 0: // fold
                return facing_bet ? std::max(0.01f, 0.8f - str * 0.8f) : 0.01f;
            case 1: // check
                return !facing_bet ? (0.3f + str * 0.2f) : 0.01f;
            case 2: // call
                return facing_bet ? std::clamp(str * 0.7f, 0.05f, 0.6f) : 0.05f;
            case 3: // raise
                return std::clamp(str * str * 0.5f, 0.01f, 0.45f);
            case 4: // allin
                return std::clamp((str - 0.8f) * 0.3f, 0.01f, 0.15f);
            default: return 0.2f;
        }
    }

    // Эксплойт-компонент: deviation от GTO в сторону pure-strategy
    static float exploit_action_prob(float str, int act_idx) noexcept {
        // Сильные руки всегда рейзят, слабые — всегда фолдят
        switch (act_idx) {
            case 0: return str < 0.3f ? 0.8f : 0.02f;  // fold
            case 1: return 0.05f;                        // check
            case 2: return str > 0.3f && str < 0.7f ? 0.7f : 0.05f; // call
            case 3: return str > 0.7f ? 0.7f : 0.05f;  // raise
            case 4: return str > 0.9f ? 0.5f : 0.02f;  // allin
            default: return 0.2f;
        }
    }
};

} // namespace baem
