#pragma once
// baem_tracker/hand_history_encoder.hpp
// HandHistoryEncoder: кодирует публичное состояние s_pub
// в вектор признаков фиксированной размерности для P_θ.
//
// Входной формат: poker::PublicState (борд, стрит, история ставок, пот)
// Выходной формат: float[FEATURE_DIM] — детерминированное кодирование
//
// Feature layout (FEATURE_DIM = 64):
//   [0..3]   — one-hot стрит (preflop/flop/turn/river)
//   [4..16]  — борд карты: rank one-hot (13 бит)
//   [17..20] — борд карты: suit mask (4 бит)
//   [21]     — pot size / 100 BB (нормализованный)
//   [22]     — current_bet / pot (относительный размер ставки)
//   [23]     — позиция: 0=OOP, 1=IP
//   [24..47] — последние 8 действий: 3 бита type + нормализованный размер (24 фичи)
//   [48..51] — агрессия за стрит (fold/check/call/raise частоты)
//   [52..63] — зарезервировано (zeros для будущих фичей)
//
// Дизайн: без heap-аллокаций, детерминированный (seed-free).
// Используется как входной вектор для PokerHistoryTransformer (Неделя 5-6).

#include "../poker_core/game_state.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace baem {

inline constexpr int FEATURE_DIM = 64;

using FeatureVec = std::array<float, FEATURE_DIM>;

class HandHistoryEncoder {
public:
    HandHistoryEncoder() noexcept = default;

    // Главный метод: кодировать s_pub → вектор признаков
    [[nodiscard]] FeatureVec encode(const poker::PublicState& spub) const noexcept {
        FeatureVec v{};   // zero-initialized

        // ── [0..3] Стрит ──────────────────────────────────────────────────
        int street_idx = stage_to_idx(spub.stage);
        v[street_idx] = 1.0f;

        // ── [4..16] Ранки борда ───────────────────────────────────────────
        spub.board.for_each([&](poker::Card c) {
            int r = c.rank();
            v[4 + r] += 1.0f / 5.0f;   // нормализуем по макс. числу карт борда
        });

        // ── [17..20] Масти борда ──────────────────────────────────────────
        spub.board.for_each([&](poker::Card c) {
            int s = c.suit();
            v[17 + s] += 0.25f;
        });

        // ── [21] Размер пота (нормализован к 100 BB) ─────────────────────
        v[21] = std::min(1.0f, static_cast<float>(spub.pot.total_bb100) / 10000.0f);

        // ── [22] Текущая ставка / пот ─────────────────────────────────────
        if (spub.pot.total_bb100 > 0) {
            v[22] = std::min(1.0f,
                static_cast<float>(spub.pot.current_bet_bb100) /
                static_cast<float>(spub.pot.total_bb100));
        }

        // ── [23] Позиция ─────────────────────────────────────────────────
        v[23] = static_cast<float>(spub.action_to_act);

        // ── [24..47] Последние 8 действий ────────────────────────────────
        int num_actions = std::min(spub.num_actions, (uint8_t)8);
        int start_idx = spub.num_actions - num_actions;
        for (int i = 0; i < num_actions; ++i) {
            const poker::Action& a = spub.action_history[start_idx + i];
            int base = 24 + i * 3;
            // Тип действия: one-hot среди {fold=0, check=1, call=2, raise/allin=3}
            int type_enc = encode_action_type(a.type);
            v[base + 0] = static_cast<float>(type_enc) / 3.0f;
            // Размер ставки нормализован
            v[base + 1] = std::min(1.0f,
                static_cast<float>(a.amount_bb100) / 10000.0f);
            // Игрок
            v[base + 2] = static_cast<float>(a.player_idx);
        }

        // ── [48..51] Агрессия за текущий стрит ───────────────────────────
        int fold_cnt=0, check_cnt=0, call_cnt=0, raise_cnt=0;
        for (int i = 0; i < spub.num_actions; ++i) {
            switch (spub.action_history[i].type) {
                case poker::ActionType::Fold:  ++fold_cnt;  break;
                case poker::ActionType::Check: ++check_cnt; break;
                case poker::ActionType::Call:  ++call_cnt;  break;
                case poker::ActionType::Raise:
                case poker::ActionType::AllIn: ++raise_cnt; break;
            }
        }
        float total_acts = static_cast<float>(std::max((uint8_t)1, spub.num_actions));
        v[48] = fold_cnt  / total_acts;
        v[49] = check_cnt / total_acts;
        v[50] = call_cnt  / total_acts;
        v[51] = raise_cnt / total_acts;

        // [52..63] — zeros (зарезервированы)
        return v;
    }

    // Batch-кодирование: массив s_pub → матрица [N × FEATURE_DIM]
    // out должен иметь размер N * FEATURE_DIM
    void encode_batch(
        const poker::PublicState* states, int N,
        float* out) const noexcept
    {
        for (int i = 0; i < N; ++i) {
            FeatureVec v = encode(states[i]);
            std::copy(v.begin(), v.end(), out + i * FEATURE_DIM);
        }
    }

    // Размерность выходного вектора
    [[nodiscard]] static constexpr int feature_dim() noexcept { return FEATURE_DIM; }

private:
    static int stage_to_idx(const poker::RoundStage& s) noexcept {
        return std::visit([](auto&& st) -> int {
            using T = std::decay_t<decltype(st)>;
            if constexpr (std::is_same_v<T, poker::Preflop>)  return 0;
            if constexpr (std::is_same_v<T, poker::Flop>)     return 1;
            if constexpr (std::is_same_v<T, poker::Turn>)     return 2;
            if constexpr (std::is_same_v<T, poker::River>)    return 3;
            return 0;
        }, s);
    }

    static int encode_action_type(poker::ActionType t) noexcept {
        switch (t) {
            case poker::ActionType::Fold:  return 0;
            case poker::ActionType::Check: return 1;
            case poker::ActionType::Call:  return 2;
            case poker::ActionType::Raise:
            case poker::ActionType::AllIn: return 3;
        }
        return 0;
    }
};

} // namespace baem
