#pragma once

#include "core/bit_packing.hpp"
#include "core/jit_cache.hpp"
#include "core/optimizers.hpp"
#include "models/pinn/pinn_model.hpp"
#include "models/sle/distillation.hpp"
#include "ucao/engine_policy.hpp"

#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace pinn {

struct RegionExample {
    std::vector<double> coords;
    double region_label = 0.0;
    double residual_score = 0.0;
    double gradient_jump = 0.0;
};

struct SleGateConfig {
    core::models::sle::DistillConfig distill{};
    bool enabled = true;
    double threshold = 0.5;
};

struct SolverStabilityNotes {
    bool sle_gate_is_differentiable = false;
    bool gradients_flow_through_gate = false;
    bool staged_training_required = true;
    const char* recommended_schedule = "train PINN backbone, distill/update SLE region gate, then fine-tune PINN with gate frozen";
};

struct HybridTrainingConfig {
    std::size_t pinn_epochs_per_stage = 16;
    std::size_t max_stages = 8;
    double boundary_error_threshold = 0.1;
    double residual_label_threshold = 0.25;
    double gradient_jump_threshold = 0.25;
    double gate_acceptance_improvement = 1e-3;
    double spatial_cell_size = 0.05;
    SleGateConfig gate{};
};

struct HybridTrainingReport {
    std::size_t completed_stages = 0;
    double best_loss = 0.0;
    double last_loss = 0.0;
    double best_gate_score = 0.0;
    bool converged = false;
    bool gate_updated = false;
};

struct SolverState {
    std::vector<std::vector<double>> parameter_values;
    core::AdamState optimizer_state{};
    core::models::sle::BooleanCascade gate_circuit{};
};

class RegionGate {
public:
    RegionGate() = default;

    void fit(const std::vector<RegionExample>& examples,
             const SleGateConfig& config = {});

    [[nodiscard]] bool ready() const noexcept { return ready_; }
    [[nodiscard]] bool classify(std::span<const double> coords) const;
    [[nodiscard]] const core::models::sle::BooleanCascade& circuit() const noexcept { return circuit_; }
    [[nodiscard]] SolverStabilityNotes stability_notes() const noexcept { return {}; }

private:
    [[nodiscard]] bool classify_packed(const std::uint64_t* packed_bits, std::size_t packed_words) const;

    mutable core::HugePageBuffer packed_workspace_{};
    core::models::sle::BooleanCascade circuit_{};
    std::vector<double> thresholds_{};
    bool ready_ = false;
};

class DiscontinuousCoefficientPDE : public PDE {
public:
    explicit DiscontinuousCoefficientPDE(std::shared_ptr<PDE> base,
                                         std::shared_ptr<RegionGate> gate,
                                         double inside_coefficient,
                                         double outside_coefficient)
        : base_(std::move(base))
        , gate_(std::move(gate))
        , inside_coefficient_(inside_coefficient)
        , outside_coefficient_(outside_coefficient) {}

    autograd::Tensor residual(const std::vector<double>& coords,
                               NeuralNetwork& net) const override;

    std::size_t input_dim() const noexcept override {
        return base_ ? base_->input_dim() : 0;
    }

    [[nodiscard]] SolverStabilityNotes stability_notes() const noexcept { return {}; }

private:
    std::shared_ptr<PDE> base_;
    std::shared_ptr<RegionGate> gate_;
    double inside_coefficient_ = 1.0;
    double outside_coefficient_ = 1.0;
};

class PINNSolver {
public:
    PINNSolver(NeuralNetwork& net, PDE& pde, Adam& optimizer)
        : model_(net, pde, optimizer), net_(net), pde_(pde), optimizer_(optimizer) {}

    void set_collocation_points(std::vector<CollocationPoint> pts) {
        collocation_points_ = pts;
        rebuild_spatial_grid();
        model_.set_collocation_points(std::move(pts));
    }
    void set_bc_points(std::vector<BCPoint> pts) { model_.set_bc_points(std::move(pts)); }
    void set_data_points(std::vector<BCPoint> pts) { model_.set_data_points(std::move(pts)); }
    void set_loss_weights(LossWeights w) { model_.set_loss_weights(w); }

    void attach_region_gate(std::shared_ptr<RegionGate> gate) { gate_ = std::move(gate); }
    [[nodiscard]] bool using_sle_region_gate() const noexcept {
        return gate_ && gate_->ready() && ucao::engine::select_runtime(ucao::engine::ModelFamily::Pinn).selected;
    }

    [[nodiscard]] SolverStabilityNotes stability_notes() const noexcept { return {}; }
    double train_step() { return model_.train_step(); }

    [[nodiscard]] std::vector<RegionExample> generate_region_examples(const HybridTrainingConfig& config) const;
    [[nodiscard]] HybridTrainingReport train_staged_hybrid(const HybridTrainingConfig& config = {});

private:
    struct GridKey {
        int x = 0;
        int y = 0;
        int z = 0;
        bool operator==(const GridKey& other) const noexcept {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct GridKeyHash {
        std::size_t operator()(const GridKey& key) const noexcept;
    };

    void rebuild_spatial_grid();
    [[nodiscard]] double evaluate_gate_score(const std::vector<RegionExample>& examples) const;
    [[nodiscard]] SolverState snapshot_state() const;
    void restore_state(const SolverState& state);
    [[nodiscard]] std::size_t nearest_neighbor_index(std::size_t anchor) const;
    [[nodiscard]] GridKey grid_key_for(std::span<const double> coords) const;

    PINNModel model_;
    NeuralNetwork& net_;
    PDE& pde_;
    Adam& optimizer_;
    std::shared_ptr<RegionGate> gate_;
    std::vector<CollocationPoint> collocation_points_;
    double spatial_cell_size_ = 0.05;
    std::unordered_map<GridKey, std::vector<std::size_t>, GridKeyHash> spatial_grid_;
};

} // namespace pinn
