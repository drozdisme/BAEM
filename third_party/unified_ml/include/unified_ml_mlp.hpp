#pragma once

#include "autograd/tensor.h"
#include "models/mlp/activation.hpp"
#include "models/mlp/loss.hpp"
#include "models/mlp/linear.hpp"
#include "models/mlp/model.hpp"
#include "models/mlp/sequential.hpp"
#include "unified_ml_dataset.hpp"
#include "unified_ml_metrics.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace unified_ml {

enum class MLPActivationKind : std::uint32_t {
    ReLU = 0,
    Sigmoid = 1,
    Tanh = 2,
};

struct MLPSpec {
    std::vector<std::size_t> hidden_layers{16, 16};
    MLPActivationKind activation = MLPActivationKind::ReLU;
    double learning_rate = 1e-3;
    std::size_t epochs = 100;
    bool binary_classification = false;
};

struct MLPFitSummary {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t epochs = 0;
    LearningTask task = LearningTask::Unknown;
    double final_loss = 0.0;
};

class UnifiedMLP {
public:
    explicit UnifiedMLP(MLPSpec spec) : spec_(std::move(spec)) {}

    [[nodiscard]] bool is_fitted() const noexcept { return static_cast<bool>(model_); }

    [[nodiscard]] MLPFitSummary fit(const DatasetView& dataset) {
        if (!dataset.has_targets()) throw std::invalid_argument("UnifiedMLP::fit: targets are required");
        if (dataset.task() != LearningTask::Classification && dataset.task() != LearningTask::Regression) {
            throw std::invalid_argument("UnifiedMLP::fit: only supervised tabular tasks are supported");
        }

        input_dim_ = dataset.cols();
        output_dim_ = 1;
        task_ = dataset.task();
        model_ = std::make_unique<mlp::Model>(build_network(input_dim_, output_dim_, spec_), spec_.learning_rate);

        double final_loss = 0.0;
        for (std::size_t epoch = 0; epoch < spec_.epochs; ++epoch) {
            final_loss = 0.0;
            for (std::size_t r = 0; r < dataset.rows(); ++r) {
                auto input = row_tensor(dataset, r);
                auto target = target_tensor(dataset.targets()[r], task_, spec_.binary_classification);
                auto loss = model_->train_step(input, target, [this](const autograd::Tensor& pred, const autograd::Tensor& target) {
                    if (task_ == LearningTask::Regression) return mlp::mse_loss(pred, target);
                    if (spec_.binary_classification) return mlp::bce_loss(pred, target);
                    return mlp::mse_loss(pred, target);
                });
                final_loss += loss.item();
            }
            final_loss /= static_cast<double>(std::max<std::size_t>(dataset.rows(), 1));
        }

        MLPFitSummary summary;
        summary.rows = dataset.rows();
        summary.cols = dataset.cols();
        summary.epochs = spec_.epochs;
        summary.task = task_;
        summary.final_loss = final_loss;
        return summary;
    }

    [[nodiscard]] PredictionSummary predict(const DatasetView& dataset, int n_classes = 0) {
        if (!model_) throw std::runtime_error("UnifiedMLP::predict: model is not fitted");

        PredictionSummary summary;
        summary.output.values.reserve(dataset.rows());
        for (std::size_t r = 0; r < dataset.rows(); ++r) {
            auto output = model_->forward(row_tensor(dataset, r));
            double value = output.item();
            if (task_ == LearningTask::Classification && spec_.binary_classification) {
                summary.output.scores.push_back(value);
                value = value >= 0.5 ? 1.0 : 0.0;
                summary.output.probabilities.push_back({1.0 - output.item(), output.item()});
            }
            summary.output.values.push_back(value);
        }

        if (dataset.has_targets()) {
            summary.evaluation = evaluate(dataset, summary.output.values, n_classes);
            summary.has_evaluation = true;
        }
        return summary;
    }

    void save(const std::string& path) const {
        if (!model_) throw std::runtime_error("UnifiedMLP::save: model is not fitted");
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("UnifiedMLP::save: unable to open file");

        const char magic[8] = {'U','M','L','M','L','P','1','\0'};
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char*>(&input_dim_), sizeof(input_dim_));
        out.write(reinterpret_cast<const char*>(&output_dim_), sizeof(output_dim_));
        out.write(reinterpret_cast<const char*>(&task_), sizeof(task_));
        out.write(reinterpret_cast<const char*>(&spec_.activation), sizeof(spec_.activation));
        out.write(reinterpret_cast<const char*>(&spec_.learning_rate), sizeof(spec_.learning_rate));
        out.write(reinterpret_cast<const char*>(&spec_.epochs), sizeof(spec_.epochs));
        out.write(reinterpret_cast<const char*>(&spec_.binary_classification), sizeof(spec_.binary_classification));

        std::size_t hidden_count = spec_.hidden_layers.size();
        out.write(reinterpret_cast<const char*>(&hidden_count), sizeof(hidden_count));
        out.write(reinterpret_cast<const char*>(spec_.hidden_layers.data()), static_cast<std::streamsize>(hidden_count * sizeof(std::size_t)));

        auto params = model_->parameters();
        std::size_t param_count = params.size();
        out.write(reinterpret_cast<const char*>(&param_count), sizeof(param_count));
        for (const auto* param : params) {
            const auto& shape = param->shape();
            std::size_t ndim = shape.size();
            out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            out.write(reinterpret_cast<const char*>(shape.data()), static_cast<std::streamsize>(ndim * sizeof(std::size_t)));
            std::size_t numel = param->numel();
            out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
            for (std::size_t i = 0; i < numel; ++i) {
                double v = param->value_flat(i);
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }
    }

    [[nodiscard]] static UnifiedMLP load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("UnifiedMLP::load: unable to open file");

        char magic[8]{};
        in.read(magic, sizeof(magic));
        const char expected[8] = {'U','M','L','M','L','P','1','\0'};
        if (std::memcmp(magic, expected, sizeof(expected)) != 0) {
            throw std::runtime_error("UnifiedMLP::load: invalid file header");
        }

        UnifiedMLP loaded(MLPSpec{});
        in.read(reinterpret_cast<char*>(&loaded.input_dim_), sizeof(loaded.input_dim_));
        in.read(reinterpret_cast<char*>(&loaded.output_dim_), sizeof(loaded.output_dim_));
        in.read(reinterpret_cast<char*>(&loaded.task_), sizeof(loaded.task_));
        in.read(reinterpret_cast<char*>(&loaded.spec_.activation), sizeof(loaded.spec_.activation));
        in.read(reinterpret_cast<char*>(&loaded.spec_.learning_rate), sizeof(loaded.spec_.learning_rate));
        in.read(reinterpret_cast<char*>(&loaded.spec_.epochs), sizeof(loaded.spec_.epochs));
        in.read(reinterpret_cast<char*>(&loaded.spec_.binary_classification), sizeof(loaded.spec_.binary_classification));

        std::size_t hidden_count = 0;
        in.read(reinterpret_cast<char*>(&hidden_count), sizeof(hidden_count));
        loaded.spec_.hidden_layers.resize(hidden_count);
        in.read(reinterpret_cast<char*>(loaded.spec_.hidden_layers.data()), static_cast<std::streamsize>(hidden_count * sizeof(std::size_t)));

        loaded.model_ = std::make_unique<mlp::Model>(build_network(loaded.input_dim_, loaded.output_dim_, loaded.spec_), loaded.spec_.learning_rate);
        auto params = loaded.model_->parameters();

        std::size_t param_count = 0;
        in.read(reinterpret_cast<char*>(&param_count), sizeof(param_count));
        if (param_count != params.size()) {
            throw std::runtime_error("UnifiedMLP::load: parameter count mismatch");
        }

        for (std::size_t p = 0; p < param_count; ++p) {
            std::size_t ndim = 0;
            in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            std::vector<std::size_t> shape(ndim, 0);
            in.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(ndim * sizeof(std::size_t)));
            std::size_t numel = 0;
            in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
            if (shape != params[p]->shape() || numel != params[p]->numel()) {
                throw std::runtime_error("UnifiedMLP::load: parameter shape mismatch");
            }
            auto data = params[p]->data();
            for (std::size_t i = 0; i < numel; ++i) {
                in.read(reinterpret_cast<char*>(&data[i]), sizeof(double));
            }
        }

        if (!in) throw std::runtime_error("UnifiedMLP::load: truncated payload");
        return loaded;
    }

private:
    static std::unique_ptr<mlp::Sequential> build_network(std::size_t input_dim,
                                                           std::size_t output_dim,
                                                           const MLPSpec& spec) {
        auto seq = std::make_unique<mlp::Sequential>();
        std::size_t current = input_dim;
        for (std::size_t hidden : spec.hidden_layers) {
            seq->add(std::make_unique<mlp::Linear>(current, hidden));
            add_activation(*seq, spec.activation);
            current = hidden;
        }
        seq->add(std::make_unique<mlp::Linear>(current, output_dim));
        if (spec.binary_classification) {
            seq->add(std::make_unique<mlp::Sigmoid>());
        }
        return seq;
    }

    static void add_activation(mlp::Sequential& seq, MLPActivationKind activation) {
        switch (activation) {
            case MLPActivationKind::ReLU: seq.add(std::make_unique<mlp::ReLU>()); break;
            case MLPActivationKind::Sigmoid: seq.add(std::make_unique<mlp::Sigmoid>()); break;
            case MLPActivationKind::Tanh: seq.add(std::make_unique<mlp::Tanh>()); break;
        }
    }

    static autograd::Tensor row_tensor(const DatasetView& dataset, std::size_t row) {
        std::vector<double> values(dataset.cols(), 0.0);
        for (std::size_t c = 0; c < dataset.cols(); ++c) values[c] = dataset.features()(row, c);
        return autograd::Tensor(std::move(values), {1, dataset.cols()}, false);
    }

    static autograd::Tensor target_tensor(double value, LearningTask task, bool binary_classification) {
        (void)task;
        (void)binary_classification;
        return autograd::Tensor(std::vector<double>{value}, {1, 1}, false);
    }

    MLPSpec spec_{};
    std::unique_ptr<mlp::Model> model_{};
    std::size_t input_dim_ = 0;
    std::size_t output_dim_ = 1;
    LearningTask task_ = LearningTask::Unknown;
};

} // namespace unified_ml
