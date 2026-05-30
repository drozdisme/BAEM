#pragma once
// baem_learning/training_batch.hpp
// TrainingBatch: кольцевой буфер опыта + построитель батчей для обучения.
//
// Experience Replay для PokerHistoryTransformer:
//   Каждый experience = (s_pub_features, observed_action, hand_outcome, lambda_hat)
//   Буфер: кольцевой, capacity = 4096 записей (настраивается)
//   Стратегия семплирования: приоритетный по abs(loss) (PER-lite)
//
// Также хранит статистику для OnlineTrainer:
//   - скользящий loss
//   - распределение наблюдаемых действий (для балансировки батча)
//   - возраст каждого experience (для decay)

#include "../baem_tracker/hand_history_encoder.hpp"
#include "../poker_core/game_state.hpp"
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace baem_learning {

using baem::FEATURE_DIM;
using baem::FeatureVec;
using baem::HandHistoryEncoder;

// ─── Одна запись опыта ────────────────────────────────────────────────────────
struct Experience {
    std::array<float, FEATURE_DIM> features{};  // HandHistoryEncoder output
    int     action{0};          // poker::ActionType as int
    float   outcome_bb{0.0f};  // результат раздачи в BB
    float   lambda_hat{0.5f};  // λ̂ в момент наблюдения
    float   priority{1.0f};    // для PER-lite (|loss| после обучения)
    uint32_t age{0};            // сколько шагов назад добавлен
    bool    valid{false};
};

// ─── Батч для обучения ─────────────────────────────────────────────────────────
struct TrainingBatch {
    std::vector<std::array<float, FEATURE_DIM>> features;
    std::vector<int>   actions;
    std::vector<float> outcomes;
    std::vector<float> lambda_hats;
    std::vector<int>   indices;    // исходные индексы в буфере (для обновления priority)
    int size{0};
};

// ─── ExperienceReplayBuffer ────────────────────────────────────────────────────
class ExperienceReplayBuffer {
public:
    static constexpr int DEFAULT_CAPACITY = 4096;
    static constexpr int MIN_BATCH_READY  = 64;  // минимум записей для начала обучения

    explicit ExperienceReplayBuffer(int capacity = DEFAULT_CAPACITY) noexcept
        : capacity_(capacity), buf_(capacity) {}

    // Добавить новый опыт
    void push(
        const FeatureVec&      features,
        poker::ActionType      action,
        float                  outcome_bb,
        float                  lambda_hat) noexcept
    {
        int idx = write_pos_ % capacity_;
        Experience& e = buf_[idx];
        e.features   = features;
        e.action     = static_cast<int>(action);
        e.outcome_bb = outcome_bb;
        e.lambda_hat = lambda_hat;
        e.priority   = 1.0f;    // начальный приоритет максимальный
        e.age        = 0;
        e.valid      = true;

        ++write_pos_;
        size_ = std::min(size_ + 1, capacity_);

        // Обновить статистику действий
        ++action_counts_[static_cast<int>(action)];
        total_pushed_++;
    }

    // Обновить приоритет записи после обучения
    void update_priority(int buf_idx, float loss) noexcept {
        if (buf_idx >= 0 && buf_idx < capacity_ && buf_[buf_idx].valid)
            buf_[buf_idx].priority = std::abs(loss) + 1e-4f;
    }

    // Построить батч размером batch_size
    // Стратегия: uniform + bias к высокому приоритету
    [[nodiscard]] TrainingBatch sample_batch(int batch_size, uint64_t seed = 0) noexcept {
        TrainingBatch batch;
        if (size_ < MIN_BATCH_READY) return batch;

        batch_size = std::min(batch_size, size_);
        batch.features.resize(batch_size);
        batch.actions.resize(batch_size);
        batch.outcomes.resize(batch_size);
        batch.lambda_hats.resize(batch_size);
        batch.indices.resize(batch_size);
        batch.size = batch_size;

        // Compute sampling weights: proportional to priority
        std::vector<float> weights(size_);
        float total_w = 0.0f;
        int base = std::max(0, write_pos_ - size_);
        for (int i = 0; i < size_; ++i) {
            weights[i] = buf_[(base + i) % capacity_].priority;
            total_w   += weights[i];
        }

        // Sample indices using categorical distribution
        std::mt19937_64 rng(seed == 0 ? static_cast<uint64_t>(write_pos_) : seed);
        std::uniform_real_distribution<float> ud(0.0f, total_w);

        for (int b = 0; b < batch_size; ++b) {
            float r = ud(rng), cum = 0.0f;
            int chosen = 0;
            for (int i = 0; i < size_; ++i) {
                cum += weights[i];
                if (r <= cum) { chosen = i; break; }
            }
            int buf_idx = (base + chosen) % capacity_;
            const Experience& e = buf_[buf_idx];
            batch.features[b]   = e.features;
            batch.actions[b]    = e.action;
            batch.outcomes[b]   = e.outcome_bb;
            batch.lambda_hats[b]= e.lambda_hat;
            batch.indices[b]    = buf_idx;
        }
        return batch;
    }

    // Полный батч из последних N записей (для gradient step)
    [[nodiscard]] TrainingBatch last_n(int n) noexcept {
        TrainingBatch batch;
        n = std::min(n, size_);
        if (n == 0) return batch;

        batch.size = n;
        batch.features.resize(n);
        batch.actions.resize(n);
        batch.outcomes.resize(n);
        batch.lambda_hats.resize(n);
        batch.indices.resize(n);

        int base = write_pos_ - n;
        for (int i = 0; i < n; ++i) {
            int buf_idx = (base + i) % capacity_;
            const Experience& e = buf_[buf_idx];
            batch.features[i]   = e.features;
            batch.actions[i]    = e.action;
            batch.outcomes[i]   = e.outcome_bb;
            batch.lambda_hats[i]= e.lambda_hat;
            batch.indices[i]    = buf_idx;
        }
        return batch;
    }

    // Возраст всех записей (для curriculum learning)
    void tick() noexcept {
        for (auto& e : buf_) if (e.valid) ++e.age;
    }

    // ── Статистика ────────────────────────────────────────────────────────────
    [[nodiscard]] int   size()         const noexcept { return size_; }
    [[nodiscard]] int   capacity()     const noexcept { return capacity_; }
    [[nodiscard]] bool  ready()        const noexcept { return size_ >= MIN_BATCH_READY; }
    [[nodiscard]] int   total_pushed() const noexcept { return total_pushed_; }

    // Частоты действий (для балансировки)
    [[nodiscard]] std::array<float, 5> action_freqs() const noexcept {
        std::array<float, 5> f{};
        int total = std::accumulate(action_counts_.begin(), action_counts_.end(), 0);
        if (total == 0) return f;
        for (int i = 0; i < 5; ++i)
            f[i] = static_cast<float>(action_counts_[i]) / total;
        return f;
    }

    void clear() noexcept {
        for (auto& e : buf_) e.valid = false;
        size_ = write_pos_ = total_pushed_ = 0;
        action_counts_.fill(0);
    }

private:
    int capacity_;
    std::vector<Experience> buf_;
    int  write_pos_{0};
    int  size_{0};
    int  total_pushed_{0};
    std::array<int, 5> action_counts_{};
};

} // namespace baem_learning
