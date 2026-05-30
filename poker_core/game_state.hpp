#pragma once
// poker_core/game_state.hpp
// Texas Hold'em state machine.
// Designed for zero-overhead abstraction:
//   - flat POD structs, no virtual functions
//   - std::variant for stage dispatch
//   - no heap allocations inside a hand
//
// Supports 2-player heads-up (easily extendable to N ≤ 9).

#include "cards.hpp"
#include <array>
#include <cstdint>
#include <variant>
#include <optional>
#include <cassert>
#include <string_view>

namespace poker {

// ─── Constants ───────────────────────────────────────────────────────────────
inline constexpr int MAX_PLAYERS = 9;
inline constexpr int MAX_ACTIONS_PER_HAND = 128;

// ─── Action ──────────────────────────────────────────────────────────────────
enum class ActionType : uint8_t {
    Fold   = 0,
    Check  = 1,
    Call   = 2,
    Raise  = 3,
    AllIn  = 4,
};

struct Action {
    ActionType type{};
    int32_t    amount_bb100{0};  // size in BB * 100 (fixed-point, avoids float)
    uint8_t    player_idx{0};

    [[nodiscard]] static constexpr std::string_view type_str(ActionType t) noexcept {
        switch (t) {
            case ActionType::Fold:  return "fold";
            case ActionType::Check: return "check";
            case ActionType::Call:  return "call";
            case ActionType::Raise: return "raise";
            case ActionType::AllIn: return "allin";
        }
        return "?";
    }
};

// ─── Round stage tags ────────────────────────────────────────────────────────
struct Preflop  {};
struct Flop     {};
struct Turn     {};
struct River    {};
struct Showdown {};

using RoundStage = std::variant<Preflop, Flop, Turn, River, Showdown>;

// ─── Per-player state ────────────────────────────────────────────────────────
struct PlayerState {
    int32_t  stack_bb100{0};    // current stack in BB*100
    int32_t  bet_bb100{0};      // amount committed this street
    Card     hole[2]{};
    bool     folded{false};
    bool     is_allin{false};
    bool     active{true};
};

// ─── Pot & betting state ─────────────────────────────────────────────────────
struct PotState {
    int32_t total_bb100{0};
    int32_t current_bet_bb100{0};  // largest bet on current street
};

// ─── PublicStateTracker (s_pub) ──────────────────────────────────────────────
// Aggregates ONLY publicly observable information:
//   - board cards, pot size, betting action sequence, current street.
// Used as the context for SelectionBias elimination (§XIV of BAEM v3 paper).
struct PublicState {
    CardSet  board{};                 // community cards
    PotState pot{};
    RoundStage stage{Preflop{}};
    uint8_t  num_players{2};
    uint8_t  dealer_pos{0};
    uint8_t  action_to_act{0};
    int8_t   hand_number{0};          // wraps, used for sequencing

    // History of public actions this hand (ring buffer)
    std::array<Action, MAX_ACTIONS_PER_HAND> action_history{};
    uint8_t  num_actions{0};

    void push_action(Action a) noexcept {
        if (num_actions < MAX_ACTIONS_PER_HAND) {
            action_history[num_actions++] = a;
        }
    }

    [[nodiscard]] bool is_terminal() const noexcept {
        return std::holds_alternative<Showdown>(stage);
    }
};

// ─── Full GameState (private + public) ───────────────────────────────────────
struct GameState {
    PublicState                         pub{};
    std::array<PlayerState, MAX_PLAYERS> players{};
    int32_t                             bb_size_chips{100};   // 1 BB = 100 chips

    // Deck (remaining cards after dealing)
    CardSet remaining_deck{full_deck()};

    void deal_hole(uint8_t player, Card c1, Card c2) noexcept {
        players[player].hole[0] = c1;
        players[player].hole[1] = c2;
        remaining_deck.remove(c1);
        remaining_deck.remove(c2);
    }

    void deal_board(Card c) noexcept {
        pub.board.add(c);
        remaining_deck.remove(c);
    }

    [[nodiscard]] PublicState& spub() noexcept       { return pub; }
    [[nodiscard]] const PublicState& spub() const noexcept { return pub; }

    // Advance to next street
    void advance_street() noexcept {
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if      constexpr (std::is_same_v<T, Preflop>)  pub.stage = Flop{};
            else if constexpr (std::is_same_v<T, Flop>)     pub.stage = Turn{};
            else if constexpr (std::is_same_v<T, Turn>)     pub.stage = River{};
            else if constexpr (std::is_same_v<T, River>)    pub.stage = Showdown{};
        }, pub.stage);

        // Reset per-street bets
        for (auto& p : players) p.bet_bb100 = 0;
        pub.pot.current_bet_bb100 = 0;
    }

    // Apply action (simplified — no side-pot logic yet)
    bool apply_action(Action a) noexcept {
        if (a.player_idx >= pub.num_players) return false;
        PlayerState& p = players[a.player_idx];

        switch (a.type) {
            case ActionType::Fold:
                p.folded = true;
                break;
            case ActionType::Check:
                // valid only if no open bet
                break;
            case ActionType::Call: {
                int32_t to_call = pub.pot.current_bet_bb100 - p.bet_bb100;
                to_call = std::min(to_call, p.stack_bb100);
                p.stack_bb100  -= to_call;
                p.bet_bb100    += to_call;
                pub.pot.total_bb100 += to_call;
                break;
            }
            case ActionType::Raise:
            case ActionType::AllIn: {
                int32_t raise_to = a.amount_bb100;
                int32_t add = raise_to - p.bet_bb100;
                add = std::min(add, p.stack_bb100);
                p.stack_bb100  -= add;
                p.bet_bb100    += add;
                pub.pot.total_bb100 += add;
                pub.pot.current_bet_bb100 = p.bet_bb100;
                if (p.stack_bb100 == 0) p.is_allin = true;
                break;
            }
        }

        pub.push_action(a);
        pub.action_to_act = (a.player_idx + 1) % pub.num_players;
        return true;
    }
};

} // namespace poker
