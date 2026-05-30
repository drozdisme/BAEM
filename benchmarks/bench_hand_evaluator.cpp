// benchmarks/bench_hand_evaluator.cpp
// HPC benchmark: HandEvaluator throughput.
// Target: > 65,000,000 ops/sec (single thread).

#include "../poker_core/poker_core.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <array>

int main() {
    using Clock = std::chrono::high_resolution_clock;
    using Sec   = std::chrono::duration<double>;

    poker::HandEvaluator eval;
    poker::DeckShuffler  deck(42);

    constexpr int N = 5'000'000;
    // Pre-generate random 7-card hands
    struct Hand7 { poker::Card c[7]; };
    std::vector<Hand7> hands(N);
    for (int i = 0; i < N; ++i) {
        deck.shuffle();
        for (int j = 0; j < 7; ++j) hands[i].c[j] = deck.deal();
    }

    // Warm-up
    volatile poker::HandStrength sink = 0;
    for (int i = 0; i < 50000; ++i)
        sink = eval.evaluate7(hands[i].c[0],hands[i].c[1],hands[i].c[2],
                              hands[i].c[3],hands[i].c[4],hands[i].c[5],hands[i].c[6]);

    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i)
        sink = eval.evaluate7(hands[i].c[0],hands[i].c[1],hands[i].c[2],
                              hands[i].c[3],hands[i].c[4],hands[i].c[5],hands[i].c[6]);
    auto t1 = Clock::now();

    double elapsed = Sec(t1 - t0).count();
    double ops_per_sec = static_cast<double>(N) / elapsed;
    bool pass = ops_per_sec >= 65e6;

    printf("HandEvaluator benchmark (evaluate7 fast path):\n");
    printf("  N           = %d hands\n", N);
    printf("  Elapsed     = %.3f s\n", elapsed);
    printf("  Throughput  = %.0f ops/sec  (%.1fM/s)\n",
           ops_per_sec, ops_per_sec/1e6);
    printf("  Target      = 65,000,000 ops/sec\n");
    printf("  Status      = %s\n", pass ? "PASS ✓" :
           "BELOW TARGET — replace hash LUT with full 2+2 for production");

    (void)sink;
    return pass ? 0 : 1;
}
