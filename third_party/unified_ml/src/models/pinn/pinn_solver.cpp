#include "models/pinn/pinn_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace pinn {
namespace {

[[nodiscard]] double mean_abs_difference(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("mean_abs_difference: size mismatch");
    if (a.empty()) return 0.0;
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += std::abs(a[i] - b[i]);
    }
    return acc / static_cast<double>(a.size());
}

} // namespace

void RegionGate::fit(const std::vector<RegionExample>& examples, const SleGateConfig& config) {
    if (examples.empty()) {
        throw std::invalid_argument("RegionGate::fit: empty region dataset");
    }

    std::vector<std::vector<double>> features;
    std::vector<double> labels;
    features.reserve(examples.size());
    labels.reserve(examples.size());
    for (const auto& ex : examples) {
        features.push_back(ex.coords);
        labels.push_back(ex.region_label);
    }

    core::models::sle::DistillDatasetBuilder builder(config.distill);
    auto dataset = builder.build(features, labels);
    auto new_circuit = core::models::sle::distill_to_circuit(dataset, config.distill);
    thresholds_ = dataset.thresholds;
    circuit_ = std::move(new_circuit);

    try {
        auto& cache = core::JitCache::instance();
        auto cached = cache.get_or_compile(circuit_);
        (void)cached;
    } catch (...) {
        // cache is opportunistic only
    }

    ready_ = true;
}

bool RegionGate::classify_packed(const std::uint64_t* packed_bits, std::size_t packed_words) const {
    if (!ready_) {
        throw std::logic_error("RegionGate::classify_packed: gate is not trained");
    }
    if (packed_bits == nullptr || packed_words == 0) {
        throw std::invalid_argument("RegionGate::classify_packed: invalid packed input buffer");
    }

    const std::size_t node_count = circuit_.input_count() + circuit_.gate_count();
    const std::size_t bytes = node_count * sizeof(std::uint64_t);
    if (packed_workspace_.size() < bytes) {
        packed_workspace_.allocate(bytes, true);
    }
    auto* nodes = reinterpret_cast<std::uint64_t*>(packed_workspace_.data());
    std::fill(nodes, nodes + node_count, 0);
    for (std::size_t i = 0; i < circuit_.input_count(); ++i) {
        nodes[i] = (i < packed_words) ? packed_bits[i] : 0;
    }

    for (std::size_t gate_idx = 0; gate_idx < circuit_.gate_count(); ++gate_idx) {
        const auto& gate = circuit_.gates()[gate_idx];
        const std::uint64_t a = nodes[gate.a];
        const std::uint64_t b = nodes[gate.b];
        const std::uint64_t c = nodes[gate.c];
        std::uint64_t out = 0;
#if defined(__AVX512F__)
        alignas(64) std::uint64_t av[8] = {a,a,a,a,a,a,a,a};
        alignas(64) std::uint64_t bv[8] = {b,b,b,b,b,b,b,b};
        alignas(64) std::uint64_t cv[8] = {c,c,c,c,c,c,c,c};
        const __m512i va = _mm512_load_epi64(av);
        const __m512i vb = _mm512_load_epi64(bv);
        const __m512i vc = _mm512_load_epi64(cv);
        const __mmask8 mask = _mm512_test_epi64_mask(_mm512_or_si512(_mm512_or_si512(va, vb), vc), _mm512_set1_epi64(-1));
        (void)mask;
#endif
        for (unsigned bit = 0; bit < 64; ++bit) {
            const unsigned idx = (((a >> bit) & 1ULL) << 2U) | (((b >> bit) & 1ULL) << 1U) | ((c >> bit) & 1ULL);
            if ((gate.mask >> idx) & 0x1U) out |= (1ULL << bit);
        }
        nodes[circuit_.input_count() + gate_idx] = out;
    }
    return (nodes[node_count - 1] & 0x1ULL) != 0;
}

bool RegionGate::classify(std::span<const double> coords) const {
    if (!ready_) {
        throw std::logic_error("RegionGate::classify: gate is not trained");
    }
    if (coords.size() != thresholds_.size()) {
        throw std::invalid_argument("RegionGate::classify: coordinate dimension mismatch");
    }

    alignas(64) std::vector<double> thresholds_aligned(thresholds_.begin(), thresholds_.end());
    alignas(64) std::vector<std::uint64_t> packed((coords.size() + 63U) / 64U, 0);
    for (std::size_t i = 0; i < coords.size(); ++i) {
        if (coords[i] >= thresholds_aligned[i]) packed[i / 64U] |= (std::uint64_t{1} << (i % 64U));
    }
    return classify_packed(packed.data(), packed.size());
}

autograd::Tensor DiscontinuousCoefficientPDE::residual(const std::vector<double>& coords,
                                                       NeuralNetwork& net) const {
    if (!base_) {
        throw std::logic_error("DiscontinuousCoefficientPDE::residual: base PDE is null");
    }
    autograd::Tensor base_residual = base_->residual(coords, net);
    if (!gate_ || !gate_->ready()) {
        return base_residual;
    }

    const bool in_region = gate_->classify(std::span<const double>(coords.data(), coords.size()));
    const double coefficient = in_region ? inside_coefficient_ : outside_coefficient_;
    return base_residual * coefficient;
}

std::size_t PINNSolver::GridKeyHash::operator()(const GridKey& key) const noexcept {
    std::size_t seed = static_cast<std::size_t>(key.x) * 73856093u;
    seed ^= static_cast<std::size_t>(key.y) * 19349663u;
    seed ^= static_cast<std::size_t>(key.z) * 83492791u;
    return seed;
}

PINNSolver::GridKey PINNSolver::grid_key_for(std::span<const double> coords) const {
    GridKey key{};
    if (!coords.empty()) key.x = static_cast<int>(std::floor(coords[0] / spatial_cell_size_));
    if (coords.size() > 1) key.y = static_cast<int>(std::floor(coords[1] / spatial_cell_size_));
    if (coords.size() > 2) key.z = static_cast<int>(std::floor(coords[2] / spatial_cell_size_));
    return key;
}

void PINNSolver::rebuild_spatial_grid() {
    spatial_grid_.clear();
    for (std::size_t i = 0; i < collocation_points_.size(); ++i) {
        spatial_grid_[grid_key_for(std::span<const double>(collocation_points_[i].coords.data(), collocation_points_[i].coords.size()))].push_back(i);
    }
}

std::size_t PINNSolver::nearest_neighbor_index(std::size_t anchor) const {
    if (collocation_points_.size() <= 1) return anchor;

    const auto key = grid_key_for(std::span<const double>(collocation_points_[anchor].coords.data(), collocation_points_[anchor].coords.size()));
    std::size_t best = anchor;
    double best_dist = std::numeric_limits<double>::infinity();

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                GridKey probe{key.x + dx, key.y + dy, key.z + dz};
                const auto it = spatial_grid_.find(probe);
                if (it == spatial_grid_.end()) continue;
                for (const auto idx : it->second) {
                    if (idx == anchor) continue;
                    double dist = 0.0;
                    const auto& src = collocation_points_[anchor].coords;
                    const auto& dst = collocation_points_[idx].coords;
                    for (std::size_t d = 0; d < std::min(src.size(), dst.size()); ++d) {
                        const double delta = src[d] - dst[d];
                        dist += delta * delta;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best = idx;
                    }
                }
            }
        }
    }

    if (best != anchor) return best;
    return anchor == 0 ? 1 : 0;
}

std::vector<RegionExample> PINNSolver::generate_region_examples(const HybridTrainingConfig& config) const {
    std::vector<RegionExample> examples;
    examples.reserve(collocation_points_.size());
    if (collocation_points_.empty()) {
        return examples;
    }

    std::vector<double> residuals;
    residuals.reserve(collocation_points_.size());
    std::vector<double> predictions;
    predictions.reserve(collocation_points_.size());

    for (const auto& pt : collocation_points_) {
        const auto residual = pde_.residual(pt.coords, net_).item();
        residuals.push_back(std::abs(residual));

        if (pt.coords.size() == 1) {
            const auto d = net_.forward_derivs_1d(pt.coords[0]);
            predictions.push_back(d.u.item());
        } else if (pt.coords.size() >= 2) {
            const auto d = net_.forward_derivs_2d(pt.coords[0], pt.coords[1]);
            predictions.push_back(d.u.item());
        } else {
            predictions.push_back(0.0);
        }
    }

    const auto packed_residuals = core::threshold_pack(std::span<const double>(residuals.data(), residuals.size()), config.residual_label_threshold);
    for (std::size_t i = 0; i < collocation_points_.size(); ++i) {
        const std::size_t nn = nearest_neighbor_index(i);
        const double gradient_jump = std::abs(predictions[i] - predictions[nn]);
        const double residual_score = residuals[i];
        const bool residual_boundary = ((packed_residuals[i / 64U] >> (i % 64U)) & 0x1U) != 0;
        const bool region_boundary = residual_boundary || gradient_jump >= config.gradient_jump_threshold;
        examples.push_back(RegionExample{
            collocation_points_[i].coords,
            region_boundary ? 1.0 : 0.0,
            residual_score,
            gradient_jump,
        });
    }
    return examples;
}

double PINNSolver::evaluate_gate_score(const std::vector<RegionExample>& examples) const {
    if (!gate_ || !gate_->ready() || examples.empty()) {
        return 0.0;
    }
    double correct = 0.0;
    for (const auto& ex : examples) {
        const bool pred = gate_->classify(std::span<const double>(ex.coords.data(), ex.coords.size()));
        const bool truth = ex.region_label >= 0.5;
        if (pred == truth) {
            correct += 1.0;
        }
    }
    return correct / static_cast<double>(examples.size());
}

SolverState PINNSolver::snapshot_state() const {
    SolverState state;
    for (auto* param : net_.parameters()) {
        state.parameter_values.push_back(static_cast<std::vector<double>>(param->data()));
    }
    state.optimizer_state = optimizer_.save_state();
    if (gate_ && gate_->ready()) {
        state.gate_circuit = gate_->circuit();
    }
    return state;
}

void PINNSolver::restore_state(const SolverState& state) {
    auto params = net_.parameters();
    for (std::size_t i = 0; i < params.size() && i < state.parameter_values.size(); ++i) {
        auto values = params[i]->data();
        const auto& src = state.parameter_values[i];
        for (std::size_t j = 0; j < src.size(); ++j) {
            values[j] = src[j];
        }
    }
    optimizer_.load_state(state.optimizer_state);
}

HybridTrainingReport PINNSolver::train_staged_hybrid(const HybridTrainingConfig& config) {
    HybridTrainingReport report{};
    spatial_cell_size_ = config.spatial_cell_size;
    rebuild_spatial_grid();

    if (!gate_) {
        gate_ = std::make_shared<RegionGate>();
    }

    double best_gate_score = gate_->ready() ? 0.0 : -1.0;
    double best_loss = std::numeric_limits<double>::infinity();

    for (std::size_t stage = 0; stage < config.max_stages; ++stage) {
        const auto state_before_stage = snapshot_state();
        for (std::size_t epoch = 0; epoch < config.pinn_epochs_per_stage; ++epoch) {
            report.last_loss = model_.train_step();
        }

        best_loss = std::min(best_loss, report.last_loss);
        const auto region_examples = generate_region_examples(config);
        if (region_examples.empty()) {
            report.completed_stages = stage + 1;
            continue;
        }

        std::vector<double> labels, preds;
        labels.reserve(region_examples.size());
        preds.reserve(region_examples.size());
        for (const auto& ex : region_examples) {
            labels.push_back(ex.region_label);
            preds.push_back(ex.residual_score);
        }
        const double boundary_error = mean_abs_difference(labels, preds);

        if (boundary_error > config.boundary_error_threshold) {
            auto candidate_gate = std::make_shared<RegionGate>();
            candidate_gate->fit(region_examples, config.gate);

            const auto previous_gate = gate_;
            gate_ = candidate_gate;
            const double candidate_score = evaluate_gate_score(region_examples);
            gate_ = previous_gate;

            if (candidate_score >= best_gate_score + config.gate_acceptance_improvement) {
                gate_ = candidate_gate;
                best_gate_score = candidate_score;
                report.gate_updated = true;
            } else {
                restore_state(state_before_stage);
            }
        }

        report.completed_stages = stage + 1;
        report.best_loss = best_loss;
        report.best_gate_score = std::max(best_gate_score, 0.0);

        if (boundary_error <= config.boundary_error_threshold &&
            report.last_loss <= best_loss + config.gate_acceptance_improvement) {
            report.converged = true;
            break;
        }
    }

    return report;
}

} // namespace pinn
