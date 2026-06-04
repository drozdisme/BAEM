// tools/train_pokerbench.cpp
// Офлайн-обучение PokerHistoryTransformer на подготовленном датасете PokerBench.
//
// Вход: текстовый файл, по одной раздаче на строку, только целые числа:
//   label street pot curbet hero_ip c1 c2 nboard [board...] nhist [type amt player]...
//   label   ∈ 0..4   (Fold,Check,Call,Raise,AllIn)
//   street  ∈ 0..3   (Preflop,Flop,Turn,River)
//   pot,curbet       — в BB*100 (фикс. точка)
//   hero_ip ∈ 0..1   (позиция героя; пишется в action_to_act)
//   c1,c2            — карманные карты героя, индекс 0..51
//   board            — карты борда, индексы 0..51
//   hist             — тройки (type 0..4, amount_bb100, player 0..1)
//
// Признаки строятся ТЕМ ЖЕ HandHistoryEncoder, что и в рантайме агента
// (encode(spub, h1, h2) — с карманными картами), поэтому веса совместимы.
//
// Сборка (без зависимостей):
//   g++ -std=c++20 -O3 -I.. train_pokerbench.cpp -o train_pokerbench
//
// Запуск:
//   ./train_pokerbench --data prepared.txt --out weights.bin --epochs 30

#include "baem_learning/poker_history_transformer.hpp"
#include "baem_tracker/hand_history_encoder.hpp"
#include "poker_core/game_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <array>
#include <chrono>

using baem::FeatureVec;
using baem::HandHistoryEncoder;
using baem_learning::PokerHistoryTransformer;
using baem_learning::PHTransformerConfig;

// Одна обучающая запись: предвычисленный вектор фич + метка действия.
struct Sample {
    FeatureVec fv;
    int label;
};

static poker::RoundStage street_to_stage(int s) {
    switch (s) {
        case 0: return poker::Preflop{};
        case 1: return poker::Flop{};
        case 2: return poker::Turn{};
        case 3: return poker::River{};
        default: return poker::Preflop{};
    }
}

// Парсит одну строку в Sample. Возвращает false при битой строке.
static bool parse_line(const std::string& line, const HandHistoryEncoder& enc, Sample& out) {
    std::istringstream is(line);
    int label, street, pot, curbet, hero_ip, c1, c2, nboard;
    if (!(is >> label >> street >> pot >> curbet >> hero_ip >> c1 >> c2 >> nboard))
        return false;
    if (label < 0 || label > 4) return false;

    poker::PublicState spub{};
    spub.stage = street_to_stage(street);
    spub.pot.total_bb100 = pot;
    spub.pot.current_bet_bb100 = curbet;
    spub.num_players = 2;
    spub.action_to_act = static_cast<uint8_t>(hero_ip ? 1 : 0);

    for (int i = 0; i < nboard; ++i) {
        int b; if (!(is >> b)) return false;
        if (b >= 0 && b < 52) spub.board.add(poker::Card{static_cast<uint8_t>(b)});
    }

    int nhist = 0;
    if (!(is >> nhist)) nhist = 0;
    for (int i = 0; i < nhist; ++i) {
        int t, amt, pl;
        if (!(is >> t >> amt >> pl)) break;
        if (t < 0 || t > 4) continue;
        poker::Action a;
        a.type = static_cast<poker::ActionType>(t);
        a.amount_bb100 = amt;
        a.player_idx = static_cast<uint8_t>(pl & 1);
        spub.push_action(a);
    }

    poker::Card h1 = (c1 >= 0 && c1 < 52) ? poker::Card{static_cast<uint8_t>(c1)} : poker::Card{};
    poker::Card h2 = (c2 >= 0 && c2 < 52) ? poker::Card{static_cast<uint8_t>(c2)} : poker::Card{};

    out.fv = enc.encode(spub, h1, h2);
    out.label = label;
    return true;
}

static const char* ACTION_NAMES[5] = {"fold", "check", "call", "raise", "allin"};

int main(int argc, char** argv) {
    std::string data_path = "prepared.txt";
    std::string out_path  = "weights.bin";
    int    epochs   = 30;
    double lr       = 3e-4;
    double val_frac = 0.1;
    uint64_t seed   = 12345;
    bool   balance  = true;   // взвешивание классов по обратной частоте
    int    patience = 5;      // ранняя остановка
    int    batch_size = 64;   // размер батча

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--data")     data_path = next();
        else if (a == "--out")      out_path  = next();
        else if (a == "--epochs")   epochs    = std::atoi(next().c_str());
        else if (a == "--lr")       lr        = std::atof(next().c_str());
        else if (a == "--val-frac") val_frac  = std::atof(next().c_str());
        else if (a == "--seed")     seed      = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--no-balance") balance = false;
        else if (a == "--patience") patience  = std::atoi(next().c_str());
        else if (a == "--batch")    batch_size = std::atoi(next().c_str());
        else if (a == "--help") {
            printf("usage: %s --data F --out F [--epochs N --lr X --val-frac X --seed N --no-balance --patience N]\n", argv[0]);
            return 0;
        }
    }

    HandHistoryEncoder enc;

    // ── Загрузка данных ──────────────────────────────────────────────────────
    printf("[load] reading %s ...\n", data_path.c_str());
    std::ifstream f(data_path);
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", data_path.c_str()); return 1; }

    std::vector<Sample> data;
    data.reserve(1 << 20);
    std::string line;
    long total = 0, bad = 0;
    std::array<long, 5> class_counts{};
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        ++total;
        Sample s;
        if (parse_line(line, enc, s)) {
            ++class_counts[s.label];
            data.push_back(std::move(s));
        } else ++bad;
    }
    printf("[load] parsed=%zu  skipped=%ld  (of %ld lines)\n", data.size(), bad, total);
    if (data.size() < 50) { fprintf(stderr, "ERROR: too few samples\n"); return 1; }

    printf("[load] class distribution:\n");
    for (int c = 0; c < 5; ++c)
        printf("        %-6s %8ld  (%.1f%%)\n", ACTION_NAMES[c], class_counts[c],
               100.0 * class_counts[c] / data.size());

    // Веса классов (обратная частота, нормированы к среднему 1.0)
    std::array<float, 5> class_w{1,1,1,1,1};
    if (balance) {
        float mean = 0; int nz = 0;
        for (int c = 0; c < 5; ++c) if (class_counts[c] > 0) {
            class_w[c] = static_cast<float>(data.size()) / (5.0f * class_counts[c]);
            mean += class_w[c]; ++nz;
        }
        mean /= std::max(1, nz);
        for (int c = 0; c < 5; ++c) class_w[c] /= mean;   // среднее ≈ 1
    }

    // ── Train/val split ──────────────────────────────────────────────────────
    std::mt19937_64 rng(seed);
    std::shuffle(data.begin(), data.end(), rng);
    size_t n_val = static_cast<size_t>(data.size() * val_frac);
    size_t n_tr  = data.size() - n_val;
    printf("[split] train=%zu  val=%zu\n", n_tr, n_val);

    // ── Модель ───────────────────────────────────────────────────────────────
    PHTransformerConfig cfg;
    cfg.lr = lr;
    PokerHistoryTransformer model(cfg);

    auto eval_val = [&](double& acc, std::array<long,5>& per_class_correct,
                        std::array<long,5>& per_class_total) {
        long correct = 0;
        per_class_correct.fill(0); per_class_total.fill(0);
        for (size_t i = n_tr; i < data.size(); ++i) {
            float p[5]; model.infer_probs(data[i].fv, p);
            int pred = static_cast<int>(std::max_element(p, p + 5) - p);
            ++per_class_total[data[i].label];
            if (pred == data[i].label) { ++correct; ++per_class_correct[data[i].label]; }
        }
        acc = n_val ? double(correct) / n_val : 0.0;
    };

    // ── Обучение ─────────────────────────────────────────────────────────────
    std::vector<size_t> order(n_tr);
    for (size_t i = 0; i < n_tr; ++i) order[i] = i;

    double best_acc = -1.0;
    int    no_improve = 0;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<FeatureVec> bfv; std::vector<int> blab; std::vector<float> bw;
    bfv.reserve(batch_size); blab.reserve(batch_size); bw.reserve(batch_size);

    for (int ep = 1; ep <= epochs; ++ep) {
        std::shuffle(order.begin(), order.end(), rng);
        double ep_loss = 0.0; int nbatch = 0;
        for (size_t k = 0; k < n_tr; ++k) {
            const Sample& s = data[order[k]];
            bfv.push_back(s.fv); blab.push_back(s.label); bw.push_back(class_w[s.label]);
            if ((int)bfv.size() == batch_size || k + 1 == n_tr) {
                ep_loss += model.train_batch_features(bfv, blab, bw);
                ++nbatch;
                bfv.clear(); blab.clear(); bw.clear();
            }
        }
        ep_loss /= std::max(1, nbatch);

        double acc; std::array<long,5> pcc, pct;
        eval_val(acc, pcc, pct);

        printf("[epoch %3d/%d] train_loss=%.4f  val_acc=%.4f", ep, epochs, ep_loss, acc);
        if (acc > best_acc + 1e-4) {
            best_acc = acc; no_improve = 0;
            model.save_weights(out_path.c_str());
            printf("  <- best (saved)\n");
        } else {
            ++no_improve;
            printf("  (no improve %d/%d)\n", no_improve, patience);
        }
        if (no_improve >= patience) { printf("[early-stop] no val improvement\n"); break; }
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Финальный отчёт на val с лучшими весами
    model.load_weights(out_path.c_str());
    double acc; std::array<long,5> pcc, pct;
    eval_val(acc, pcc, pct);
    printf("\n=== DONE ===\n");
    printf("best val accuracy : %.4f\n", acc);
    printf("train time        : %.1f s\n", secs);
    printf("weights saved to  : %s\n", out_path.c_str());
    printf("per-class recall (val):\n");
    for (int c = 0; c < 5; ++c)
        printf("   %-6s %6ld/%-6ld  %.3f\n", ACTION_NAMES[c], pcc[c], pct[c],
               pct[c] ? double(pcc[c]) / pct[c] : 0.0);
    return 0;
}
