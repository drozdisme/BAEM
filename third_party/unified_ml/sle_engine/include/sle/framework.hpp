#pragma once

#include "sle/full_engine.hpp"

#include <string>
#include <vector>

namespace sle {

enum class TaskKind {
    BooleanFunction,
    ConstrainedBooleanFunction,
    StochasticBitstream,
};

struct Sample {
    std::vector<BitVector> inputs;
    BitVector target;
};

class Dataset {
public:
    Dataset() = default;
    Dataset(std::vector<Sample> samples, TaskKind kind = TaskKind::BooleanFunction)
        : samples_(std::move(samples)), kind_(kind) {}

    [[nodiscard]] const std::vector<Sample>& samples() const noexcept { return samples_; }
    [[nodiscard]] TaskKind kind() const noexcept { return kind_; }
    [[nodiscard]] bool empty() const noexcept { return samples_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }

    [[nodiscard]] std::vector<TrainingExample> as_training_examples() const;

private:
    std::vector<Sample> samples_;
    TaskKind kind_ = TaskKind::BooleanFunction;
};

struct TrainerConfig {
    FullEngineConfig engine;
    bool enable_topology_mutation = true;
    bool enable_hlc_penalty = true;
};

struct Metrics {
    double output_fitness = 0.0;
    double total_fitness = 0.0;
    std::size_t errors = 0;
};

class FrameworkModel {
public:
    FrameworkModel() = default;
    explicit FrameworkModel(FullEngineModel model) : model_(std::move(model)) {}

    [[nodiscard]] const FullEngineModel& raw() const noexcept { return model_; }
    [[nodiscard]] BitVector predict(const std::vector<BitVector>& inputs) const;

private:
    FullEngineModel model_;
};

class Trainer {
public:
    explicit Trainer(TrainerConfig config) : config_(std::move(config)) {}

    [[nodiscard]] FrameworkModel fit(const Dataset& dataset, const HardLogicContract& contract = {});
    [[nodiscard]] Metrics evaluate(const FrameworkModel& model,
                                   const Dataset& dataset,
                                   const HardLogicContract& contract = {}) const;

private:
    TrainerConfig config_;
};

[[nodiscard]] Dataset make_boolean_dataset(const std::vector<TrainingExample>& examples,
                                           TaskKind kind = TaskKind::BooleanFunction);

[[nodiscard]] TrainerConfig default_trainer_config_for_task(TaskKind kind);

} // namespace sle
