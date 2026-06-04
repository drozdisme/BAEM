// tools/play_pokerbench.cpp
// Инференс обученного агента.
//
// Режим 1 (eval): метрики на подготовленном файле
//   ./play_pokerbench --weights weights.bin --eval prepared_test.txt
//
// Режим 2 (play): рекомендация по одному сценарию (та же строка-формат, label игнорируется)
//   ./play_pokerbench --weights weights.bin --scenario one_line.txt
//   echo "0 1 600 200 1 51 50 3 12 25 38 1 3 200 0" | ./play_pokerbench --weights w.bin --stdin
//
// Формат строки идентичен train_pokerbench (label в режиме play игнорируется,
// можно ставить 0).

#include "baem_learning/poker_history_transformer.hpp"
#include "baem_tracker/hand_history_encoder.hpp"
#include "poker_core/game_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <algorithm>

using baem::FeatureVec;
using baem::HandHistoryEncoder;
using baem_learning::PokerHistoryTransformer;

static const char* ACTION_NAMES[5] = {"fold", "check", "call", "raise", "allin"};

static poker::RoundStage street_to_stage(int s) {
    switch (s) { case 1: return poker::Flop{}; case 2: return poker::Turn{};
                 case 3: return poker::River{}; default: return poker::Preflop{}; }
}

static bool parse_line(const std::string& line, const HandHistoryEncoder& enc,
                       FeatureVec& fv, int& label) {
    std::istringstream is(line);
    int street, pot, curbet, hero_ip, c1, c2, nboard;
    if (!(is >> label >> street >> pot >> curbet >> hero_ip >> c1 >> c2 >> nboard)) return false;
    poker::PublicState spub{};
    spub.stage = street_to_stage(street);
    spub.pot.total_bb100 = pot;
    spub.pot.current_bet_bb100 = curbet;
    spub.num_players = 2;
    spub.action_to_act = static_cast<uint8_t>(hero_ip ? 1 : 0);
    for (int i = 0; i < nboard; ++i) { int b; if (!(is >> b)) return false;
        if (b >= 0 && b < 52) spub.board.add(poker::Card{static_cast<uint8_t>(b)}); }
    int nhist = 0; if (is >> nhist) for (int i = 0; i < nhist; ++i) {
        int t, amt, pl; if (!(is >> t >> amt >> pl)) break;
        poker::Action a; a.type = static_cast<poker::ActionType>(t & 7);
        a.amount_bb100 = amt; a.player_idx = static_cast<uint8_t>(pl & 1); spub.push_action(a); }
    poker::Card h1 = (c1>=0&&c1<52)?poker::Card{static_cast<uint8_t>(c1)}:poker::Card{};
    poker::Card h2 = (c2>=0&&c2<52)?poker::Card{static_cast<uint8_t>(c2)}:poker::Card{};
    fv = enc.encode(spub, h1, h2);
    return true;
}

int main(int argc, char** argv) {
    std::string weights = "weights.bin", eval_path, scenario_path;
    bool use_stdin = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i+1<argc)?argv[++i]:""; };
        if      (a == "--weights")  weights = next();
        else if (a == "--eval")     eval_path = next();
        else if (a == "--scenario") scenario_path = next();
        else if (a == "--stdin")    use_stdin = true;
    }

    HandHistoryEncoder enc;
    PokerHistoryTransformer model;
    if (!model.load_weights(weights.c_str())) {
        fprintf(stderr, "ERROR: cannot load weights %s\n", weights.c_str()); return 1;
    }

    auto recommend = [&](const FeatureVec& fv) {
        float p[5]; model.infer_probs(fv, p);
        int best = static_cast<int>(std::max_element(p, p + 5) - p);
        printf("recommended action: %s  (p=%.3f)\n", ACTION_NAMES[best], p[best]);
        printf("full policy: ");
        for (int c = 0; c < 5; ++c) printf("%s=%.3f ", ACTION_NAMES[c], p[c]);
        printf("\n");
    };

    // ── Режим eval ───────────────────────────────────────────────────────────
    if (!eval_path.empty()) {
        std::ifstream f(eval_path);
        if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", eval_path.c_str()); return 1; }
        long correct = 0, n = 0;
        std::array<long,5> pcc{}, pct{};
        std::string line; FeatureVec fv; int label;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (!parse_line(line, enc, fv, label)) continue;
            float p[5]; model.infer_probs(fv, p);
            int pred = static_cast<int>(std::max_element(p, p+5) - p);
            ++n; ++pct[label]; if (pred == label) { ++correct; ++pcc[label]; }
        }
        printf("eval on %ld samples\n", n);
        printf("accuracy: %.4f\n", n ? double(correct)/n : 0.0);
        for (int c = 0; c < 5; ++c)
            printf("   %-6s recall %6ld/%-6ld  %.3f\n", ACTION_NAMES[c], pcc[c], pct[c],
                   pct[c]?double(pcc[c])/pct[c]:0.0);
        return 0;
    }

    // ── Режим play ───────────────────────────────────────────────────────────
    std::string line;
    if (use_stdin) {
        if (!std::getline(std::cin, line)) { fprintf(stderr,"no input\n"); return 1; }
    } else if (!scenario_path.empty()) {
        std::ifstream f(scenario_path);
        if (!f || !std::getline(f, line)) { fprintf(stderr,"cannot read scenario\n"); return 1; }
    } else {
        fprintf(stderr, "specify --eval F | --scenario F | --stdin\n"); return 1;
    }
    FeatureVec fv; int label;
    if (!parse_line(line, enc, fv, label)) { fprintf(stderr,"bad scenario line\n"); return 1; }
    recommend(fv);
    return 0;
}
