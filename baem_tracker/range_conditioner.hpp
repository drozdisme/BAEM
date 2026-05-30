#pragma once
// baem_tracker/range_conditioner.hpp
// RangeConditioner: управляет допустимым диапазоном карт соперника.
//
// Отвечает за:
//   1. Начальную инициализацию диапазона из 1326 комбо (C(52,2))
//   2. Блокировку комбо при открытии карты борда  
//   3. Блокировку комбо при известных картах агента (hole cards агента)
//   4. Блокировку при вскрытии карт соперника (showdown)
//   5. Перенормировку после каждой блокировки
//
// Хранит вероятности в SoA-раскладке (Structure of Arrays) для
// эффективной AVX-512 векторизации в BeliefTracker.

#include "../poker_core/cards.hpp"
#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <span>

namespace baem {

// ─── ComboIndex ─────────────────────────────────────────────────────────────
// Уникальный индекс пары карт [0..1325], i < j → idx = i*52 - i*(i+1)/2 + j - i - 1
// Но используем более простой подход: линейный индекс по паре (i,j), i<j
inline constexpr int combo_index(int i, int j) noexcept {
    // i < j, возвращает номер в лексикографическом порядке
    return i * 52 - (i * (i + 1) / 2) + (j - i - 1);
}

// Итоговый размер: C(52,2) = 1326
inline constexpr int COMBO_COUNT = 1326;

// ─── SoA пробабилити-буфер ───────────────────────────────────────────────────
// Разделяем карты и вероятности для векторизации:
//   c1_idx[], c2_idx[]  — uint8_t массивы индексов карт
//   prob[]              — float32 массив вероятностей (aligned 64 bytes)
//   active[]            — bool маска допустимых комбо
struct RangeBuffer {
    std::vector<uint8_t>  c1_idx;   // индекс первой карты [0..51]
    std::vector<uint8_t>  c2_idx;   // индекс второй карты [0..51]
    std::vector<float>    prob;     // вероятность B_t(h), Σ=1
    std::vector<uint8_t>  active;   // 1 = комбо допустимо, 0 = заблокировано

    int capacity{0};   // полный размер (включая неактивные)
    int num_active{0}; // число активных комбо

    void resize(int n) {
        c1_idx.resize(n, 0);
        c2_idx.resize(n, 0);
        prob.resize(n, 0.0f);
        active.resize(n, 1);
        capacity = n;
        num_active = n;
    }

    void clear() {
        c1_idx.clear(); c2_idx.clear();
        prob.clear(); active.clear();
        capacity = num_active = 0;
    }
};

// ─── RangeConditioner ────────────────────────────────────────────────────────
class RangeConditioner {
public:
    // Инициализация диапазона: все 1326 комбо, равномерный prior.
    // known_dead: карты, гарантированно отсутствующие у соперника
    //   (борд + hole cards агента)
    void init(poker::CardSet known_dead = poker::CardSet{}) noexcept {
        buf_.clear();
        buf_.resize(COMBO_COUNT);
        known_dead_ = known_dead;

        int idx = 0;
        int alive = 0;
        for (int i = 0; i < 52; ++i) {
            for (int j = i + 1; j < 52; ++j, ++idx) {
                poker::Card ci{static_cast<uint8_t>(i)};
                poker::Card cj{static_cast<uint8_t>(j)};
                buf_.c1_idx[idx] = static_cast<uint8_t>(i);
                buf_.c2_idx[idx] = static_cast<uint8_t>(j);
                bool blocked = known_dead.contains(ci) || known_dead.contains(cj);
                buf_.active[idx] = blocked ? 0 : 1;
                buf_.prob[idx] = 0.0f;
                if (!blocked) ++alive;
            }
        }
        buf_.num_active = alive;

        // Равномерный prior по активным комбо
        float unif = alive > 0 ? 1.0f / static_cast<float>(alive) : 0.0f;
        for (int k = 0; k < COMBO_COUNT; ++k)
            buf_.prob[k] = buf_.active[k] ? unif : 0.0f;
    }

    // Добавить карту борда / hole card агента — блокирует соответствующие комбо
    void block_card(poker::Card c) noexcept {
        if (known_dead_.contains(c)) return;
        known_dead_.add(c);

        int deactivated = 0;
        for (int k = 0; k < COMBO_COUNT; ++k) {
            if (!buf_.active[k]) continue;
            if (buf_.c1_idx[k] == c.idx || buf_.c2_idx[k] == c.idx) {
                buf_.active[k] = 0;
                buf_.prob[k]   = 0.0f;
                ++deactivated;
            }
        }
        buf_.num_active -= deactivated;
        renormalize();
    }

    // Блокировать несколько карт за раз (вскрытие борда)
    void block_cards(poker::CardSet cards) noexcept {
        cards.for_each([this](poker::Card c){ block_card(c); });
    }

    // Принудительно задать диапазон: только комбо с этими двумя картами
    // (при вскрытии карт соперника на шоудауне)
    void condition_on_showdown(poker::Card c1, poker::Card c2) noexcept {
        int target = find_combo(c1.idx, c2.idx);
        for (int k = 0; k < COMBO_COUNT; ++k) {
            if (k == target) {
                buf_.active[k] = 1;
                buf_.prob[k]   = 1.0f;
            } else {
                buf_.active[k] = 0;
                buf_.prob[k]   = 0.0f;
            }
        }
        buf_.num_active = (target >= 0) ? 1 : 0;
    }

    // Прямой доступ к буферу (для BeliefTracker и бенчмарков)
    [[nodiscard]] RangeBuffer&       buffer()       noexcept { return buf_; }
    [[nodiscard]] const RangeBuffer& buffer() const noexcept { return buf_; }
    [[nodiscard]] int  num_active()            const noexcept { return buf_.num_active; }
    [[nodiscard]] poker::CardSet known_dead()  const noexcept { return known_dead_; }

    // Проверка нормировки (для тестов)
    [[nodiscard]] float prob_sum() const noexcept {
        float s = 0.0f;
        for (int k = 0; k < COMBO_COUNT; ++k) s += buf_.prob[k];
        return s;
    }

    // Вернуть вероятность конкретного комбо
    [[nodiscard]] float prob_of(poker::Card c1, poker::Card c2) const noexcept {
        int k = find_combo(c1.idx, c2.idx);
        return (k >= 0) ? buf_.prob[k] : 0.0f;
    }

    // Применить лайклихуды извне (для стыковки с BeliefTracker)
    // lh[k] = P_θ(a | h_k) — вектор размером COMBO_COUNT
    // Неактивные комбо игнорируются.
    void apply_likelihood(std::span<const float> lh) noexcept {
        assert(static_cast<int>(lh.size()) == COMBO_COUNT);
        float sum = 0.0f;
        for (int k = 0; k < COMBO_COUNT; ++k) {
            if (!buf_.active[k]) { buf_.prob[k] = 0.0f; continue; }
            buf_.prob[k] *= lh[k];
            sum += buf_.prob[k];
        }
        if (sum > 1e-30f) {
            float inv = 1.0f / sum;
            for (int k = 0; k < COMBO_COUNT; ++k) buf_.prob[k] *= inv;
        } else {
            // Degeneracy: reset active combos to uniform
            float unif = buf_.num_active > 0
                ? 1.0f / static_cast<float>(buf_.num_active) : 0.0f;
            for (int k = 0; k < COMBO_COUNT; ++k)
                buf_.prob[k] = buf_.active[k] ? unif : 0.0f;
        }
    }

    // Сброс к uniform prior без изменения маски known_dead
    void reset_to_uniform() noexcept {
        float unif = buf_.num_active > 0
            ? 1.0f / static_cast<float>(buf_.num_active) : 0.0f;
        for (int k = 0; k < COMBO_COUNT; ++k)
            buf_.prob[k] = buf_.active[k] ? unif : 0.0f;
    }

private:
    RangeBuffer   buf_{};
    poker::CardSet known_dead_{};

    void renormalize() noexcept {
        float s = 0.0f;
        for (int k = 0; k < COMBO_COUNT; ++k) s += buf_.prob[k];
        if (s > 1e-30f) {
            float inv = 1.0f / s;
            for (int k = 0; k < COMBO_COUNT; ++k) buf_.prob[k] *= inv;
        } else {
            reset_to_uniform();
        }
    }

    // Найти индекс комбо по двум индексам карт (порядок неважен)
    [[nodiscard]] static int find_combo(int a, int b) noexcept {
        if (a > b) std::swap(a, b);
        // Линейный поиск — вызывается редко (только на showdown)
        int idx = 0;
        for (int i = 0; i < 52; ++i)
            for (int j = i + 1; j < 52; ++j, ++idx)
                if (i == a && j == b) return idx;
        return -1;
    }
};

} // namespace baem
