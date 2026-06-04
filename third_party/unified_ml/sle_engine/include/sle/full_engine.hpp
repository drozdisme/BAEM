#pragma once

#include "sle/contracts.hpp"
#include "sle/encoding.hpp"
#include "sle/synthesis.hpp"

#include <optional>

namespace sle {

struct FullEngineConfig {
    EncodingMode encoding_mode = EncodingMode::BinaryDirect;
    std::size_t stream_length = 64;
    std::size_t gate_count = 3;
    bool hlc_aware = true;
    SynthesisConfig synthesis{};
    SolverPolicy solver_policy{};
    ResidualPlannerConfig residual_planner{};
};

struct FullEngineModel {
    BooleanCascade base;
    std::optional<BooleanCascade> residual;
    HardLogicContract contract;
    FullEngineConfig config;
};

struct FullEngineTrainResult {
    FullEngineModel model;
    FitnessBreakdown fitness;
    SynthesisTier selected_tier = SynthesisTier::Local;
    std::vector<TierDiagnostics> tier_diagnostics;
    std::string selection_summary;
};

[[nodiscard]] FullEngineTrainResult train_full_engine(const std::vector<TrainingExample>& dataset,
                                                      const FullEngineConfig& config,
                                                      HardLogicContract contract = {});

[[nodiscard]] BitVector run_full_engine(const FullEngineModel& model,
                                        const std::vector<BitVector>& inputs);

} // namespace sle
