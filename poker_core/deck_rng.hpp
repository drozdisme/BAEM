#pragma once
// poker_core/deck_rng.hpp
// Thread-safe deck shuffler.
// Uses xoshiro256** — extremely fast, passes PractRand.
// Each thread owns its own RNG state (no contention).
// Interface mirrors unified_ml::core::random where available.

#include "cards.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <thread>

namespace poker {

// ─── xoshiro256** PRNG ────────────────────────────────────────────────────────
struct Xoshiro256ss {
    std::array<uint64_t, 4> s{};

    explicit Xoshiro256ss(uint64_t seed) noexcept {
        // splitmix64 seeding
        uint64_t z = seed;
        for (auto& x : s) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            x = z ^ (z >> 31);
        }
    }

    [[nodiscard]] uint64_t next() noexcept {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Unbiased integer in [0, n)
    [[nodiscard]] uint64_t next_bounded(uint64_t n) noexcept {
        uint64_t threshold = (~n + 1) % n;
        while (true) {
            uint64_t r = next();
            if (r >= threshold) return r % n;
        }
    }

private:
    [[nodiscard]] static uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
};

// ─── DeckShuffler ─────────────────────────────────────────────────────────────
// Maintains a 52-card deck, shuffles via Fisher-Yates,
// deals cards without replacement.
class DeckShuffler {
public:
    explicit DeckShuffler(uint64_t seed = 0) noexcept
        : rng_(seed == 0 ? default_seed() : seed)
    {
        for (int i = 0; i < 52; ++i) deck_[i] = Card{static_cast<uint8_t>(i)};
    }

    void shuffle() noexcept {
        for (int i = 51; i > 0; --i) {
            int j = static_cast<int>(rng_.next_bounded(static_cast<uint64_t>(i + 1)));
            Card tmp = deck_[i]; deck_[i] = deck_[j]; deck_[j] = tmp;
        }
        top_ = 0;
    }

    [[nodiscard]] Card deal() noexcept {
        assert(top_ < 52);
        return deck_[top_++];
    }

    [[nodiscard]] Card peek(int offset = 0) const noexcept {
        assert(top_ + offset < 52);
        return deck_[top_ + offset];
    }

    [[nodiscard]] int cards_left() const noexcept { return 52 - top_; }

    // Deal n cards into an output span
    void deal_n(std::span<Card> out) noexcept {
        for (auto& c : out) c = deal();
    }

private:
    Xoshiro256ss rng_;
    std::array<Card, 52> deck_{};
    int top_{52};  // initialized past end; caller must shuffle first

    [[nodiscard]] static uint64_t default_seed() noexcept {
        // Mix thread id and timestamp for decent uniqueness
        auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return static_cast<uint64_t>(tid) ^ 0xdeadbeefcafeULL;
    }
};

} // namespace poker
