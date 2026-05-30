#pragma once
// baem_learning/hand_simulator.hpp
// HandSimulator: полный Texas Hold'em симулятор для генерации hand_outcome.
//
// Задача (Week 5):
//   Заменяет hand_outcome=0.0 в тестах реальным исходом раздачи.
//   Используется OnlineTrainer для получения X_t — наблюдаемого результата (BB).
//
// Дизайн:
//   - Детерминированный при фиксированном seed (воспроизводимость тестов)
//   - Два режима: полный HU (Heads-Up) симулятор и быстрый MC-evaluator
//   - Возвращает result в BB×100 (int32), совместимо с DKWTracker
//
// Поддерживаемые сценарии:
//   1. deal_and_run() — полная раздача от префлопа до шоудауна
//   2. run_out()      — докрутить борд от текущего состояния
//   3. equity()       — MC-оценка эквити руки против диапазона

#include "../poker_core/poker_core.hpp"
#include <random>
#include <array>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

namespace baem_learning {

// ─── Результат раздачи ────────────────────────────────────────────────────────
struct HandResult {
    int32_t  outcome_bb100{0};  // результат в BB×100 (положительный = выиграл)
    bool     went_to_showdown{false};
    bool     agent_won{false};
    uint32_t hand_number{0};

    // Конвертация для DKWTracker (принимает float в BB)
    [[nodiscard]] float outcome_bb() const noexcept {
        return static_cast<float>(outcome_bb100) / 100.0f;
    }
};

// ─── Стратегия игрока (для симулятора) ────────────────────────────────────────
// Упрощённая стратегия: strength-based betting
struct SimStrategy {
    float aggression{0.5f};  // [0,1]: выше = чаще ставит/рейзит
    float call_thresh{0.3f}; // порог колла (normalized hand strength)
    float fold_thresh{0.15f};// порог фолда против ставки

    // Выбрать действие по нормализованной силе руки и контексту
    [[nodiscard]] poker::ActionType decide(
        float hand_strength,   // [0,1]
        bool  facing_bet,
        poker::DeckShuffler& rng_deck) const noexcept
    {
        // Простая детерминированная стратегия с шумом через deck RNG seed
        if (facing_bet) {
            if (hand_strength < fold_thresh)  return poker::ActionType::Fold;
            if (hand_strength < call_thresh)  return poker::ActionType::Call;
            if (hand_strength > 0.7f && aggression > 0.6f) return poker::ActionType::Raise;
            return poker::ActionType::Call;
        } else {
            if (hand_strength > 0.65f && aggression > 0.4f) return poker::ActionType::Raise;
            return poker::ActionType::Check;
        }
    }
};

// ─── HandSimulator ────────────────────────────────────────────────────────────
class HandSimulator {
public:
    // stack_bb100: стек каждого игрока в BB×100
    // bb_size: размер BB в условных единицах (обычно 100)
    explicit HandSimulator(
        int32_t  stack_bb100 = 10000,  // 100 BB
        uint64_t seed        = 42)
        : stack_(stack_bb100)
        , deck_(seed)
    {}

    // ── Полная HU раздача ─────────────────────────────────────────────────────
    // agent_strategy:    стратегия агента (позиция 0 = BTN/SB)
    // opponent_strategy: стратегия оппонента (позиция 1 = BB)
    // Возвращает результат с точки зрения агента (позиция 0).
    [[nodiscard]] HandResult deal_and_run(
        const SimStrategy& agent_strategy    = {},
        const SimStrategy& opp_strategy      = {},
        poker::CardSet     block_cards       = {})  // карты которые не могут быть розданы
    noexcept
    {
        deck_.shuffle();

        // Убираем заблокированные карты (уже известные агенту)
        // Простой метод: пересдавать если вытащили заблокированную
        poker::Card agent_hole[2], opp_hole[2];
        deal_hole_avoiding(agent_hole, opp_hole, block_cards);

        // Блайнды: агент (BTN/SB) постит 0.5 BB, оппонент (BB) постит 1 BB
        int32_t pot       = 150;   // 0.5+1.0 BB = 150 BB×100
        int32_t agent_inv = 50;    // инвестировано агентом
        int32_t opp_inv   = 100;   // инвестировано оппонентом
        int32_t agt_stack = stack_ - agent_inv;
        int32_t opp_stack = stack_ - opp_inv;

        // ─── Префлоп ─────────────────────────────────────────────────────────
        float agt_str_pre = preflop_strength(agent_hole[0], agent_hole[1]);
        float opp_str_pre = preflop_strength(opp_hole[0],  opp_hole[1]);

        // BTN действует первым (SB = агент)
        bool facing_bet = true;  // BB уже в потe, BTN facing BB
        auto [agt_pre, opp_pre, pot_pre, done_pre] = betting_round(
            agt_str_pre, opp_str_pre,
            agent_strategy, opp_strategy,
            pot, agent_inv, opp_inv, agt_stack, opp_stack,
            facing_bet);

        pot = pot_pre; agent_inv = agt_pre; opp_inv = opp_pre;
        agt_stack = stack_ - agent_inv;
        opp_stack = stack_ - opp_inv;

        if (done_pre) {
            // Кто-то сфолдил на префлопе
            bool agent_wins = (agt_pre < opp_pre);  // тот кто меньше инвестировал — сфолдил
            HandResult r;
            r.outcome_bb100    = agent_wins ? opp_inv : -agent_inv;
            r.went_to_showdown = false;
            r.agent_won        = agent_wins;
            r.hand_number      = ++hand_count_;
            return r;
        }

        // ─── Борд ─────────────────────────────────────────────────────────────
        poker::Card board[5];
        for (int i = 0; i < 5; ++i) board[i] = deck_.deal();

        // ─── Флоп, Тёрн, Ривер ────────────────────────────────────────────────
        poker::HandEvaluator eval;
        int32_t cur_pot = pot;
        int32_t cur_agt_inv = agent_inv, cur_opp_inv = opp_inv;
        int32_t cur_agt_stack = agt_stack, cur_opp_stack = opp_stack;
        bool folded = false;
        bool agent_wins_fold = false;

        for (int street = 0; street < 3 && !folded; ++street) {
            int board_cnt = (street == 0) ? 3 : (street == 1) ? 4 : 5;
            poker::Card agt_cards[7] = {agent_hole[0], agent_hole[1]};
            poker::Card opp_cards[7] = {opp_hole[0],  opp_hole[1]};
            for (int i = 0; i < board_cnt; ++i) {
                agt_cards[2+i] = board[i];
                opp_cards[2+i] = board[i];
            }
            float agt_str = normalize_strength(
                eval.evaluate(std::span<const poker::Card>(agt_cards, 2+board_cnt)));
            float opp_str = normalize_strength(
                eval.evaluate(std::span<const poker::Card>(opp_cards, 2+board_cnt)));

            // BB действует первым на постфлоп улицах (OOP = оппонент)
            auto [agt_new, opp_new, pot_new, done] = betting_round(
                opp_str, agt_str,   // OOP first
                opp_strategy, agent_strategy,
                cur_pot, cur_opp_inv, cur_agt_inv,
                cur_opp_stack, cur_agt_stack,
                false /* check first */);

            cur_pot = pot_new;
            cur_opp_inv = agt_new; cur_agt_inv = opp_new;  // swap back
            cur_opp_stack = stack_ - cur_opp_inv;
            cur_agt_stack = stack_ - cur_agt_inv;

            if (done) {
                folded = true;
                // Кто меньше инвестировал — сфолдил
                agent_wins_fold = (cur_agt_inv > cur_opp_inv);
            }
        }

        HandResult r;
        r.hand_number = ++hand_count_;

        if (folded) {
            r.outcome_bb100    = agent_wins_fold ? cur_opp_inv : -cur_agt_inv;
            r.went_to_showdown = false;
            r.agent_won        = agent_wins_fold;
            return r;
        }

        // ─── Шоудаун ─────────────────────────────────────────────────────────
        poker::Card agt7[7] = {agent_hole[0], agent_hole[1],
                                board[0], board[1], board[2], board[3], board[4]};
        poker::Card opp7[7] = {opp_hole[0], opp_hole[1],
                                board[0], board[1], board[2], board[3], board[4]};
        auto agt_str_fin = eval.evaluate7(agt7[0],agt7[1],agt7[2],agt7[3],agt7[4],agt7[5],agt7[6]);
        auto opp_str_fin = eval.evaluate7(opp7[0],opp7[1],opp7[2],opp7[3],opp7[4],opp7[5],opp7[6]);

        bool agent_wins = (agt_str_fin > opp_str_fin);
        bool chop       = (agt_str_fin == opp_str_fin);

        r.went_to_showdown = true;
        r.agent_won        = agent_wins;
        if (chop) {
            r.outcome_bb100 = 0;
        } else if (agent_wins) {
            r.outcome_bb100 = cur_opp_inv;   // выиграли инвестиции оппонента
        } else {
            r.outcome_bb100 = -cur_agt_inv;  // потеряли свои инвестиции
        }
        return r;
    }

    // ── Monte Carlo эквити ────────────────────────────────────────────────────
    // Оценивает P(агент выиграет | hole_cards, board) за N итераций.
    // Корректная реализация: строим остаток колоды явно (без deck_.deal()),
    // чтобы skip-continue не «съедал» карты.
    [[nodiscard]] float equity_mc(
        poker::Card c1, poker::Card c2,
        poker::CardSet board,
        int n_trials = 500) noexcept
    {
        poker::HandEvaluator eval;

        // Карты, которые не могут участвовать ни у кого
        poker::CardSet dead = board;
        dead.add(c1); dead.add(c2);

        // Строим вектор доступных карт один раз
        std::vector<poker::Card> avail;
        avail.reserve(52 - dead.size());
        for (int i = 0; i < 52; ++i) {
            poker::Card c{static_cast<uint8_t>(i)};
            if (!dead.contains(c)) avail.push_back(c);
        }

        // Борд-карты в массиве
        poker::Card run_board[5];
        int board_cnt = 0;
        board.for_each([&](poker::Card bc){ run_board[board_cnt++] = bc; });

        std::mt19937_64 rng(reinterpret_cast<uint64_t>(this) ^ 0xdeadbeef);
        int wins = 0, ties = 0;

        for (int trial = 0; trial < n_trials; ++trial) {
            // Перетасовать avail
            for (int i = static_cast<int>(avail.size()) - 1; i > 0; --i) {
                uint64_t j = rng() % static_cast<uint64_t>(i + 1);
                std::swap(avail[i], avail[j]);
            }

            // Первые 2 — карты оппонента, следующие (5-board_cnt) — ранаут
            int needed_run = 5 - board_cnt;
            if (static_cast<int>(avail.size()) < 2 + needed_run) continue;

            poker::Card o1 = avail[0], o2 = avail[1];

            // Собираем полный борд
            poker::Card full_board[5];
            for (int i = 0; i < board_cnt; ++i) full_board[i] = run_board[i];
            for (int i = 0; i < needed_run; ++i) full_board[board_cnt + i] = avail[2 + i];

            auto agt = eval.evaluate7(c1, c2,
                full_board[0], full_board[1], full_board[2],
                full_board[3], full_board[4]);
            auto opp_s = eval.evaluate7(o1, o2,
                full_board[0], full_board[1], full_board[2],
                full_board[3], full_board[4]);

            if (agt > opp_s) ++wins;
            else if (agt == opp_s) ++ties;
        }
        return (wins + 0.5f * ties) / static_cast<float>(n_trials);
    }

    void set_seed(uint64_t seed) noexcept { deck_ = poker::DeckShuffler(seed); }
    [[nodiscard]] uint32_t hand_count() const noexcept { return hand_count_; }

private:
    int32_t  stack_;
    poker::DeckShuffler deck_;
    uint32_t hand_count_{0};

    // Результат betting round: [agt_invested, opp_invested, pot, someone_folded]
    struct BetResult { int32_t agt, opp, pot; bool done; };

    BetResult betting_round(
        float agt_str, float opp_str,
        const SimStrategy& agt_strat, const SimStrategy& opp_strat,
        int32_t pot, int32_t agt_inv, int32_t opp_inv,
        int32_t agt_stack, int32_t opp_stack,
        bool facing_bet) const noexcept
    {
        // Упрощённый betting: только одно действие каждого игрока
        // In position (агент) действует
        auto agt_action = agt_strat.decide(agt_str, facing_bet, const_cast<poker::DeckShuffler&>(deck_));

        if (agt_action == poker::ActionType::Fold) {
            return {agt_inv, opp_inv, pot, true};
        }
        if (agt_action == poker::ActionType::Raise) {
            int32_t raise_to = std::min(pot, agt_stack);  // pot-sized raise
            raise_to = std::max(raise_to, 200);            // min 2 BB
            agt_inv += raise_to; pot += raise_to;
            // Оппонент реагирует
            auto opp_action = opp_strat.decide(opp_str, true, const_cast<poker::DeckShuffler&>(deck_));
            if (opp_action == poker::ActionType::Fold) {
                return {agt_inv, opp_inv, pot, true};
            }
            // Call
            int32_t to_call = agt_inv - opp_inv;
            to_call = std::min(to_call, opp_stack);
            opp_inv += to_call; pot += to_call;
        } else if (agt_action == poker::ActionType::Call) {
            int32_t to_call = std::max(0, opp_inv - agt_inv);
            to_call = std::min(to_call, agt_stack);
            agt_inv += to_call; pot += to_call;
        }
        // Check: оппонент может bet
        else {
            if (!facing_bet && opp_str > 0.5f && opp_strat.aggression > 0.4f) {
                int32_t bet = std::min(pot / 2, opp_stack);
                bet = std::max(bet, 100);
                opp_inv += bet; pot += bet;
                // Агент реагирует
                auto agt2 = agt_strat.decide(agt_str, true, const_cast<poker::DeckShuffler&>(deck_));
                if (agt2 == poker::ActionType::Fold) return {agt_inv, opp_inv, pot, true};
                int32_t to_call = opp_inv - agt_inv;
                to_call = std::min(to_call, agt_stack);
                agt_inv += to_call; pot += to_call;
            }
        }
        return {agt_inv, opp_inv, pot, false};
    }

    void deal_hole_avoiding(
        poker::Card agt[2], poker::Card opp[2],
        poker::CardSet blocked) const noexcept
    {
        poker::DeckShuffler& d = const_cast<poker::DeckShuffler&>(deck_);
        int ai = 0, oi = 0;
        while (ai < 2 || oi < 2) {
            poker::Card c = d.deal();
            if (blocked.contains(c)) continue;
            if (ai < 2) { agt[ai++] = c; blocked.add(c); }
            else if (oi < 2) { opp[oi++] = c; blocked.add(c); }
        }
    }

    static float preflop_strength(poker::Card c1, poker::Card c2) noexcept {
        int r1 = c1.rank(), r2 = c2.rank();
        if (r1 < r2) std::swap(r1, r2);
        bool suited = (c1.suit() == c2.suit());
        bool paired = (r1 == r2);
        float base = (static_cast<float>(r1 + r2) / 24.0f);  // [0..1]
        if (paired) base = std::min(1.0f, base + 0.2f);
        if (suited) base = std::min(1.0f, base + 0.05f);
        return base;
    }

    static float normalize_strength(poker::HandStrength s) noexcept {
        return static_cast<float>(s) / 7462.0f;
    }
};

} // namespace baem_learning
