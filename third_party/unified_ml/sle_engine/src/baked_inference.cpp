#include "sle/baked_inference.hpp"

#include "sle/topology_cache.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sle {

namespace {

constexpr std::uint32_t kBakedArtifactMagic = 0x534C4542; // BELS
constexpr std::uint32_t kBakedArtifactVersion = 1;

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("failed to read baked artifact field");
    return value;
}

void write_string(std::ofstream& out, const std::string& value) {
    const auto size = static_cast<std::uint64_t>(value.size());
    write_pod(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string read_string(std::ifstream& in) {
    const auto size = read_pod<std::uint64_t>(in);
    std::string value(size, '\0');
    in.read(value.data(), static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("failed to read baked artifact string");
    return value;
}

void write_bytes(std::ofstream& out, const std::vector<std::uint8_t>& bytes) {
    const auto size = static_cast<std::uint64_t>(bytes.size());
    write_pod(out, size);
    if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> read_bytes(std::ifstream& in) {
    const auto size = read_pod<std::uint64_t>(in);
    std::vector<std::uint8_t> bytes(size);
    if (size != 0) in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("failed to read baked artifact bytes");
    return bytes;
}

void write_cascade(std::ofstream& out, const BooleanCascade& cascade) {
    write_pod(out, static_cast<std::uint64_t>(cascade.input_count()));
    write_pod(out, static_cast<std::uint64_t>(cascade.gate_count()));
    for (const auto& gate : cascade.gates()) {
        write_pod(out, static_cast<std::uint64_t>(gate.a));
        write_pod(out, static_cast<std::uint64_t>(gate.b));
        write_pod(out, static_cast<std::uint64_t>(gate.c));
        write_pod(out, gate.mask);
    }
}

BooleanCascade read_cascade(std::ifstream& in) {
    const auto input_count = read_pod<std::uint64_t>(in);
    const auto gate_count = read_pod<std::uint64_t>(in);
    BooleanCascade cascade(input_count);
    for (std::uint64_t i = 0; i < gate_count; ++i) {
        TernaryGate gate;
        gate.a = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
        gate.b = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
        gate.c = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
        gate.mask = read_pod<std::uint8_t>(in);
        cascade.add_gate(gate);
    }
    return cascade;
}

void write_patch_points(std::ofstream& out, const std::vector<GatePatchPoint>& patch_points) {
    write_pod(out, static_cast<std::uint64_t>(patch_points.size()));
    for (const auto& point : patch_points) {
        write_pod(out, static_cast<std::uint64_t>(point.gate_index));
        write_pod(out, static_cast<std::uint64_t>(point.immediate_offset));
    }
}

std::vector<GatePatchPoint> read_patch_points(std::ifstream& in) {
    const auto count = read_pod<std::uint64_t>(in);
    std::vector<GatePatchPoint> patch_points;
    patch_points.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        GatePatchPoint point;
        point.gate_index = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
        point.immediate_offset = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
        patch_points.push_back(point);
    }
    return patch_points;
}

} // namespace

BakedInferenceArtifact bake_inference_artifact(const BooleanCascade& cascade,
                                               const BakeConfig& config) {
    auto compiled = compile_boolean_cascade(cascade, config.jit);

    BakedInferenceArtifact artifact;
    artifact.signature = topology_signature(cascade);
    artifact.artifact_kind = BakedArtifactKind::BooleanCascadeJit;
    artifact.pattern_kind = ExactPatternKind::None;
    artifact.target_instruction_set = target_instruction_set_for_backend(compiled.backend());
    artifact.backend = compiled.backend();
    artifact.input_count = compiled.input_count();
    artifact.gate_count = compiled.gate_count();
    artifact.workspace_word_count = compiled.workspace_word_count();
    artifact.cascade = cascade;
    artifact.patch_points = compiled.patch_points();
    artifact.code_bytes.assign(compiled.raw_code(), compiled.raw_code() + compiled.code_size());
    artifact.compiled = std::make_shared<CompiledBooleanCascade>(std::move(compiled));
    artifact.prepared = true;
    return artifact;
}

BakedInferenceArtifact bake_exact_pattern_artifact(const ExactPatternMatch& match,
                                                   const BakeConfig& config) {
    auto artifact = bake_inference_artifact(match.cascade, config);
    artifact.pattern_kind = match.kind;
    return artifact;
}

BakedInferenceArtifact bake_branchless_decision_artifact(const BranchlessDecisionArtifactSpec& spec,
                                                         const BakeConfig& config) {
    BakedInferenceArtifact artifact;
    artifact.signature = "branchless-decision:p" + std::to_string(spec.predicate_count) + ":l" + std::to_string(spec.leaf_count);
    artifact.artifact_kind = BakedArtifactKind::BranchlessDecisionLogic;
    artifact.pattern_kind = ExactPatternKind::None;
    artifact.target_instruction_set = target_instruction_set_for_backend(config.jit.preferred_backend);
    artifact.backend = config.jit.preferred_backend;
    artifact.input_count = spec.predicate_count;
    artifact.branchless_program = spec.program;
    artifact.prepared = true;
    return artifact;
}

void save_baked_inference_artifact(const BakedInferenceArtifact& artifact, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open baked artifact for writing");

    write_pod(out, kBakedArtifactMagic);
    write_pod(out, kBakedArtifactVersion);
    write_pod(out, static_cast<std::uint32_t>(artifact.artifact_kind));
    write_pod(out, static_cast<std::uint32_t>(artifact.pattern_kind));
    write_pod(out, static_cast<std::uint32_t>(artifact.target_instruction_set));
    write_pod(out, static_cast<std::uint32_t>(artifact.backend));
    write_pod(out, static_cast<std::uint64_t>(artifact.input_count));
    write_pod(out, static_cast<std::uint64_t>(artifact.gate_count));
    write_pod(out, static_cast<std::uint64_t>(artifact.workspace_word_count));
    write_string(out, artifact.signature);
    write_cascade(out, artifact.cascade);
    write_patch_points(out, artifact.patch_points);
    write_bytes(out, artifact.code_bytes);
    write_bytes(out, artifact.branchless_program);

    if (!out) throw std::runtime_error("failed to write baked artifact");
}

BakedInferenceArtifact load_baked_inference_artifact(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open baked artifact for reading");

    const auto magic = read_pod<std::uint32_t>(in);
    const auto version = read_pod<std::uint32_t>(in);
    if (magic != kBakedArtifactMagic || version != kBakedArtifactVersion) {
        throw std::runtime_error("unsupported baked artifact format");
    }

    BakedInferenceArtifact artifact;
    artifact.artifact_kind = static_cast<BakedArtifactKind>(read_pod<std::uint32_t>(in));
    artifact.pattern_kind = static_cast<ExactPatternKind>(read_pod<std::uint32_t>(in));
    artifact.target_instruction_set = static_cast<TargetInstructionSet>(read_pod<std::uint32_t>(in));
    artifact.backend = static_cast<JitBackend>(read_pod<std::uint32_t>(in));
    artifact.input_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    artifact.gate_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    artifact.workspace_word_count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    artifact.signature = read_string(in);
    artifact.cascade = read_cascade(in);
    artifact.patch_points = read_patch_points(in);
    artifact.code_bytes = read_bytes(in);
    artifact.branchless_program = read_bytes(in);
    artifact.prepared = false;
    return artifact;
}

void prepare_inference_runtime(BakedInferenceArtifact& artifact) {
    if (!runtime_supports_instruction_set(artifact.target_instruction_set)) {
        throw std::runtime_error("instruction set mismatch during artifact preparation");
    }
    if (artifact.artifact_kind == BakedArtifactKind::BranchlessDecisionLogic) {
        artifact.prepared = true;
        return;
    }
    if (artifact.compiled == nullptr) {
        artifact.compiled = rehydrate_compiled_boolean_cascade(
            artifact.input_count,
            artifact.gate_count,
            artifact.workspace_word_count,
            artifact.backend,
            artifact.patch_points,
            artifact.code_bytes);
    }
    artifact.prepared = artifact.compiled != nullptr && artifact.compiled->valid();
}

InferenceResult run_baked_inference(const BakedInferenceArtifact& artifact,
                                    const std::vector<BitVector>& inputs,
                                    const InferencePolicy& policy) {
    if (policy.mode == ExecutionMode::InferenceOnly) {
        if (!artifact.prepared || artifact.compiled == nullptr) {
            if (artifact.artifact_kind == BakedArtifactKind::BranchlessDecisionLogic && artifact.prepared) {
                if (inputs.size() != artifact.input_count) {
                    return InferenceResult{false, InferenceError::InputShapeMismatch, {}};
                }
                return InferenceResult{true, InferenceError::PatternNotFound, inputs.empty() ? BitVector{} : inputs.front()};
            }
            return InferenceResult{false, InferenceError::ArtifactNotPrepared, {}};
        }
        if (!runtime_supports_instruction_set(artifact.target_instruction_set)) {
            return InferenceResult{false, InferenceError::InstructionSetMismatch, {}};
        }
        if (inputs.size() != artifact.input_count) {
            return InferenceResult{false, InferenceError::InputShapeMismatch, {}};
        }
        if (artifact.artifact_kind == BakedArtifactKind::BranchlessDecisionLogic) {
            if (inputs.size() != artifact.input_count) {
                return InferenceResult{false, InferenceError::InputShapeMismatch, {}};
            }
            return InferenceResult{true, InferenceError::PatternNotFound, inputs.empty() ? BitVector{} : inputs.front()};
        }
        return InferenceResult{true, InferenceError::PatternNotFound, artifact.compiled->evaluate(inputs)};
    }

    if (!artifact.prepared || artifact.compiled == nullptr) {
        if (artifact.artifact_kind == BakedArtifactKind::BranchlessDecisionLogic && artifact.prepared) {
            if (inputs.size() != artifact.input_count) {
                return InferenceResult{false, InferenceError::InputShapeMismatch, {}};
            }
            return InferenceResult{true, InferenceError::PatternNotFound, inputs.empty() ? BitVector{} : inputs.front()};
        }
        return InferenceResult{false, InferenceError::ArtifactNotPrepared, {}};
    }
    return InferenceResult{true, InferenceError::PatternNotFound, artifact.compiled->evaluate(inputs)};
}

InferenceResolution resolve_inference_only_artifact(const BooleanCascade& initial,
                                                    const std::vector<TrainingExample>& dataset,
                                                    const BakeConfig& config,
                                                    CompilationCache* cache) {
    InferenceResolution resolution;
    if (config.inference_policy.mode != ExecutionMode::InferenceOnly) {
        resolution.error = InferenceError::PatternNotFound;
        resolution.summary = "resolver requires inference-only mode";
        return resolution;
    }

    const auto signature = topology_signature(initial);
    if (cache != nullptr && config.inference_policy.allow_cache_lookup && cache->contains(signature)) {
        auto cached = cache->get(signature);
        resolution.found = true;
        resolution.error = InferenceError::PatternNotFound;
        resolution.summary = "resolved from compilation cache";
        resolution.artifact.signature = cached.signature;
        resolution.artifact.artifact_kind = BakedArtifactKind::BooleanCascadeJit;
        resolution.artifact.target_instruction_set = target_instruction_set_for_backend(cached.compiled->backend());
        resolution.artifact.backend = cached.compiled->backend();
        resolution.artifact.input_count = cached.compiled->input_count();
        resolution.artifact.gate_count = cached.compiled->gate_count();
        resolution.artifact.workspace_word_count = cached.compiled->workspace_word_count();
        resolution.artifact.cascade = initial;
        resolution.artifact.patch_points = cached.compiled->patch_points();
        resolution.artifact.code_bytes.assign(cached.compiled->raw_code(), cached.compiled->raw_code() + cached.compiled->code_size());
        resolution.artifact.compiled = cached.compiled;
        resolution.artifact.prepared = true;
        return resolution;
    }

    if (config.inference_policy.allow_exact_registry) {
        auto patterns = exact_pattern_registry(initial, dataset, initial.gate_count());
        if (!patterns.empty()) {
            resolution.found = true;
            resolution.error = InferenceError::PatternNotFound;
            resolution.summary = "resolved from exact pattern registry";
            resolution.artifact = bake_exact_pattern_artifact(patterns.front(), config);
            return resolution;
        }
    }

    resolution.error = InferenceError::PatternNotFound;
    resolution.summary = "pattern not found in cache or exact registry";
    return resolution;
}

} // namespace sle
