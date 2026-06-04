#pragma once

#include "sle/jit.hpp"
#include "sle/synthesis.hpp"
#include "sle/topology_cache.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sle {

enum class InferenceError {
    PatternNotFound,
    ArtifactNotPrepared,
    BackendUnavailable,
    InstructionSetMismatch,
    InputShapeMismatch,
    SerializationError,
};

enum class BakedArtifactKind {
    BooleanCascadeJit,
    BranchlessDecisionLogic,
};

enum class ExecutionMode {
    SynthesisAllowed,
    InferenceOnly,
};

struct InferencePolicy {
    ExecutionMode mode = ExecutionMode::SynthesisAllowed;
    bool allow_cache_lookup = true;
    bool allow_exact_registry = true;
    bool allow_search = true;
};

struct BakeConfig {
    JitCompileOptions jit{};
    InferencePolicy inference_policy{};
};

struct BakedInferenceArtifact {
    std::string signature;
    BakedArtifactKind artifact_kind = BakedArtifactKind::BooleanCascadeJit;
    ExactPatternKind pattern_kind = ExactPatternKind::None;
    TargetInstructionSet target_instruction_set = TargetInstructionSet::ScalarX86_64;
    JitBackend backend = JitBackend::ScalarGpr;
    std::size_t input_count = 0;
    std::size_t gate_count = 0;
    std::size_t workspace_word_count = 0;
    BooleanCascade cascade;
    std::vector<GatePatchPoint> patch_points;
    std::vector<std::uint8_t> code_bytes;
    std::vector<std::uint8_t> branchless_program;
    std::shared_ptr<CompiledBooleanCascade> compiled;
    bool prepared = false;
};

struct InferenceResolution {
    bool found = false;
    InferenceError error = InferenceError::PatternNotFound;
    BakedInferenceArtifact artifact;
    std::string summary;
};

struct BranchlessDecisionArtifactSpec {
    std::size_t predicate_count = 0;
    std::size_t leaf_count = 0;
    std::vector<std::uint8_t> program;
};

struct InferenceResult {
    bool ok = false;
    InferenceError error = InferenceError::PatternNotFound;
    BitVector output;
};

[[nodiscard]] BakedInferenceArtifact bake_inference_artifact(const BooleanCascade& cascade,
                                                             const BakeConfig& config = {});
[[nodiscard]] BakedInferenceArtifact bake_exact_pattern_artifact(const ExactPatternMatch& match,
                                                                 const BakeConfig& config = {});
[[nodiscard]] BakedInferenceArtifact bake_branchless_decision_artifact(const BranchlessDecisionArtifactSpec& spec,
                                                                       const BakeConfig& config = {});
void save_baked_inference_artifact(const BakedInferenceArtifact& artifact, const std::string& path);
[[nodiscard]] BakedInferenceArtifact load_baked_inference_artifact(const std::string& path);
void prepare_inference_runtime(BakedInferenceArtifact& artifact);
[[nodiscard]] InferenceResult run_baked_inference(const BakedInferenceArtifact& artifact,
                                                  const std::vector<BitVector>& inputs,
                                                  const InferencePolicy& policy = {});
[[nodiscard]] InferenceResolution resolve_inference_only_artifact(
    const BooleanCascade& initial,
    const std::vector<TrainingExample>& dataset,
    const BakeConfig& config,
    CompilationCache* cache = nullptr);

} // namespace sle
