#pragma once
// poker_core/cards.hpp
// Card representation using 64-bit bitboards.
// Layout: bits 0..51 → cards in rank-major order (2c=0, 3c=1, …, Ac=12,
//         2d=13, …, Ad=25, 2h=26, …, Ah=38, 2s=39, …, As=51).
// No dynamic allocations; all types are trivially copyable.

#include <cstdint>
#include <cstdlib>
#include <array>
#include <bit>
#include <cassert>
#include <string_view>

namespace poker {

// ─── Constants ───────────────────────────────────────────────────────────────
inline constexpr int NUM_CARDS   = 52;
inline constexpr int NUM_RANKS   = 13;  // 2..A
inline constexpr int NUM_SUITS   = 4;   // c d h s

// ─── Card ────────────────────────────────────────────────────────────────────
struct Card {
    // index ∈ [0, 51]
    uint8_t idx{255};  // 255 = null / unknown

    [[nodiscard]] constexpr bool valid() const noexcept { return idx < 52; }
    [[nodiscard]] constexpr int  rank()  const noexcept { return idx % 13; } // 0=2 … 12=A
    [[nodiscard]] constexpr int  suit()  const noexcept { return idx / 13; } // 0=c 1=d 2=h 3=s

    static constexpr Card from_rank_suit(int r, int s) noexcept {
        return Card{static_cast<uint8_t>(s * 13 + r)};
    }

    constexpr bool operator==(const Card& o) const noexcept { return idx == o.idx; }
    constexpr bool operator!=(const Card& o) const noexcept { return idx != o.idx; }

    [[nodiscard]] static constexpr std::string_view rank_str(int r) noexcept {
        constexpr std::string_view tbl[13] = {
            "2","3","4","5","6","7","8","9","T","J","Q","K","A"
        };
        return tbl[r];
    }
    [[nodiscard]] static constexpr std::string_view suit_str(int s) noexcept {
        constexpr std::string_view tbl[4] = {"c","d","h","s"};
        return tbl[s];
    }
};

// ─── CardSet (bitboard) ──────────────────────────────────────────────────────
// Represents a set of cards as a 64-bit mask.
struct CardSet {
    uint64_t mask{0};

    constexpr void add(Card c) noexcept    { mask |=  (uint64_t{1} << c.idx); }
    constexpr void remove(Card c) noexcept { mask &= ~(uint64_t{1} << c.idx); }
    [[nodiscard]] constexpr bool contains(Card c) const noexcept {
        return (mask >> c.idx) & 1;
    }
    [[nodiscard]] constexpr int  size()   const noexcept { return std::popcount(mask); }
    [[nodiscard]] constexpr bool empty()  const noexcept { return mask == 0; }

    // Iterate set bits
    template<typename Fn>
    constexpr void for_each(Fn&& fn) const noexcept {
        uint64_t m = mask;
        while (m) {
            int i = std::countr_zero(m);
            fn(Card{static_cast<uint8_t>(i)});
            m &= m - 1;
        }
    }

    [[nodiscard]] constexpr CardSet operator|(const CardSet& o) const noexcept {
        return CardSet{mask | o.mask};
    }
    [[nodiscard]] constexpr CardSet operator&(const CardSet& o) const noexcept {
        return CardSet{mask & o.mask};
    }
    [[nodiscard]] constexpr CardSet operator~() const noexcept {
        // Only first 52 bits are valid
        constexpr uint64_t FULL = (uint64_t{1} << 52) - 1;
        return CardSet{(~mask) & FULL};
    }
};

// ─── Full deck ───────────────────────────────────────────────────────────────
[[nodiscard]] inline constexpr CardSet full_deck() noexcept {
    constexpr uint64_t FULL = (uint64_t{1} << 52) - 1;
    return CardSet{FULL};
}

} // namespace poker
