#include "sle/engine.hpp"
#include "sle/baked_inference.hpp"
#include "sle/fast_ops.hpp"
#include "sle/framework.hpp"
#include "sle/full_engine.hpp"
#include "sle/jit.hpp"
#include "sle/jit_batch.hpp"
#include "sle/jit_trainer.hpp"
#include "sle/perception.hpp"
#include "sle/synthesis.hpp"
#include "sle/topology_search.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char* name) {
    if (ok) {
        std::cout << "PASS  " << name << '\n';
        ++g_pass;
    } else {
        std::cout << "FAIL  " << name << '\n';
        ++g_fail;
    }
}

sle::BitVector from_string(const char* bits) {
    const std::string s(bits);
    sle::BitVector v(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) v.set(i, s[i] == '1');
    return v;
}

void test_ternary_xor() {
    auto a = from_string("01011010");
    auto b = from_string("00111100");
    auto c = from_string("00001111");
    auto out = sle::ternary_apply(a, b, c, 0x96);
    check(out.to_string() == "01101001", "ternary 3-input xor mask 0x96");
}

void test_mixer_invertibility() {
    auto matrix = sle::GF2ByteMatrix::from_row_xors({{1, 0}, {2, 1}, {7, 3}});
    check(matrix.is_invertible(), "gf2 row-xor generated matrix invertible");

    sle::BitMixer mixer(matrix);
    auto input = from_string("10110010");
    auto mixed = mixer.mix(input);
    auto restored = mixer.unmix(mixed);
    check(restored.to_string() == input.to_string(), "gf2 mixer inverse restores input");
}

void test_residual_exactness() {
    auto base = from_string("10101010");
    auto target = from_string("11001100");
    auto residue = sle::residual_target(target, base);
    auto output = sle::residual_apply(base, residue);
    check(output.to_string() == target.to_string(), "residual xor exactness");
}

void test_forbidden_filter() {
    auto value = from_string("11110000");
    auto forbidden = from_string("00110011");
    auto filtered = sle::apply_forbidden_filter(value, forbidden);
    check(filtered.to_string() == "11000000", "forbidden filter safety");
}

void test_required_verification() {
    auto value = from_string("11010010");
    auto required = from_string("01010000");
    auto bad = from_string("01010100");
    check(sle::required_ok(value, required), "required verification accepts valid output");
    check(!sle::required_ok(required, bad), "required verification rejects missing bit");
}

void test_engine_path() {
    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x96});
    sle::HardLogicContract contract{from_string("00000000"), from_string("00000001")};
    sle::Engine engine(cascade, sle::BitMixer{}, contract);

    auto out = engine.run({from_string("10101010"), from_string("11001100"), from_string("11110000")});
    check(out.to_string() == "10010110", "engine pipeline mixer->cascade->contract");
}

void test_compiled_cascade_matches_interpreter() {
    auto a = from_string("10101010110011001111000000001111");
    auto b = from_string("11110000000011110101010100110011");
    auto c = from_string("00001111111100001010101011001100");

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x96});
    cascade.add_gate({3, 1, 0, 0xE8});

    const auto interpreted = cascade.evaluate({a, b, c});
    const auto compiled = sle::compile_boolean_cascade(cascade).evaluate({a, b, c});
    check(interpreted.to_string() == compiled.to_string(), "register-level JIT matches interpreter");
}

void test_jit_patch_in_place_binary() {
    auto a = from_string("10101010");
    auto b = from_string("11001100");
    auto c = from_string("11110000");

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x96});

    sle::JitCompileOptions options;
    options.preferred_backend = sle::JitBackend::Avx512Ternlog;
    auto compiled = sle::compile_boolean_cascade(cascade, options);
    if (compiled.backend() != sle::JitBackend::Avx512Ternlog) {
        check(true, "jit patch-in-place scalar skipped without avx512 patch backend");
        return;
    }
    auto before = compiled.evaluate({a, b, c});
    compiled.patch_gate_mask(0, 0xE8);
    auto after = compiled.evaluate({a, b, c});

    check(before.to_string() == sle::ternary_apply(a, b, c, 0x96).to_string(), "jit binary before patch is correct");
    check(after.to_string() == sle::ternary_apply(a, b, c, 0xE8).to_string(), "jit patch-in-place updates binary");
}

void test_jit_first_trainer_finds_xor() {
    if (!sle::runtime_supports_jit_backend(sle::JitBackend::Avx512Ternlog)) {
        check(true, "jit-first trainer skipped without patchable backend");
        return;
    }
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x00});

    sle::SynthesisConfig cfg;
    cfg.iterations = 2048;
    cfg.prefer_jit = true;

    auto result = sle::synthesize_local(cascade, dataset, cfg);
    check(result.cascade.gates().front().mask == 0x96, "jit-first trainer finds xor mask");
    check(result.fitness.output_fitness == 1.0, "jit-first trainer reaches perfect xor fitness");
}

void test_prepare_batch_transposes_bitplanes() {
    auto bitvec = [](const char* bits) {
        return from_string(bits);
    };

    std::vector<sle::TrainingExample> dataset{
        {{bitvec("10"), bitvec("01")}, bitvec("11")},
        {{bitvec("01"), bitvec("10")}, bitvec("00")},
        {{bitvec("11"), bitvec("11")}, bitvec("10")},
    };

    auto batch = sle::prepare_batch(dataset, 64, 64);
    check(batch.example_count == 3, "prepare_batch preserves example count");
    check(batch.padded_examples == 64, "prepare_batch pads to register width");
    check((batch.input_ptrs[0][0] & 0x7ULL) == 0x5ULL, "prepare_batch transposes first input bit-plane");
    check((batch.input_ptrs[1][0] & 0x7ULL) == 0x6ULL, "prepare_batch transposes second input bit-plane");
}

void test_jit_trainer_selection_and_expansion() {
    auto bitvec = [](const char* bits) { return from_string(bits); };
    std::vector<sle::TrainingExample> dataset{
        {{bitvec("10"), bitvec("01"), bitvec("11")}, bitvec("11")},
        {{bitvec("01"), bitvec("10"), bitvec("00")}, bitvec("00")},
        {{bitvec("11"), bitvec("11"), bitvec("10")}, bitvec("10")},
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x00});

    auto evaluator = sle::make_safe_batched_jit_evaluator(dataset, 64, 64);
    sle::SynthesisConfig cfg;
    cfg.prefer_jit = false;
    if (!sle::runtime_supports_jit_backend(sle::JitBackend::Avx512Ternlog)) {
        check(true, "jit trainer selection skipped without patchable backend");
        return;
    }

    sle::JITTrainer trainer(cascade, evaluator, cfg, sle::JitCompileOptions{8, sle::JitBackend::Avx512Ternlog});

    auto signal = trainer.compute_selection_signal();
    auto masks = trainer.expand_masks(0, signal.error_gradient);
    check(!masks.empty(), "jit trainer expansion returns candidates");
}

void test_mdl_prefers_shorter_program() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
    };

    sle::BooleanCascade short_cascade(3);
    short_cascade.add_gate({0, 1, 2, 0xF0});

    sle::BooleanCascade long_cascade(3);
    long_cascade.add_gate({0, 1, 2, 0xF0});
    long_cascade.add_gate({3, 1, 2, 0xF0});

    sle::SynthesisConfig cfg;
    cfg.use_mdl = true;

    const auto short_fit = sle::evaluate_fitness(short_cascade, dataset, cfg);
    const auto long_fit = sle::evaluate_fitness(long_cascade, dataset, cfg);
    check(short_fit.mdl_model_bits < long_fit.mdl_model_bits, "mdl estimates shorter hypothesis as cheaper");
}

void test_fast_matches_scalar() {
    auto a = from_string("10101010110011001111000000001111");
    auto b = from_string("11110000000011110101010100110011");
    auto c = from_string("00001111111100001010101011001100");
    auto scalar = sle::ternary_apply(a, b, c, 0xCA);
    auto fast = sle::ternary_apply_fast(a, b, c, 0xCA);
    check(scalar.to_string() == fast.to_string(), "fast ternary path matches scalar");

    auto matrix = sle::GF2ByteMatrix::from_row_xors({{1, 0}, {3, 2}, {4, 1}, {7, 6}});
    sle::BitMixer mixer(matrix);
    auto mixed_fast = mixer.mix(a);
    auto scalar_lut = sle::apply_byte_map_fast(a, sle::build_byte_lut(matrix.rows()));
    check(mixed_fast.to_string() == scalar_lut.to_string(), "fast mixer path stable");
}

void test_mcts_synthesis_xor() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x00});

    sle::SynthesisConfig cfg;
    cfg.iterations = 512;
    auto diagnostics = sle::SearchDiagnostics{};
    auto result = sle::synthesize_mcts(cascade, dataset, cfg, &diagnostics);
    check(result.cascade.gates().front().mask == 0x96, "mcts synthesis finds xor mask");
    check(result.fitness.output_fitness == 1.0, "mcts synthesis reaches perfect output fitness");
    check(diagnostics.expanded_nodes > 0, "mcts synthesis emits diagnostics");
}

void test_local_greedy_synthesis_xor() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x00});

    sle::SynthesisConfig cfg;
    cfg.iterations = 512;
    auto diagnostics = sle::SearchDiagnostics{};
    auto result = sle::synthesize_local_greedy(cascade, dataset, cfg, &diagnostics);
    bool exact = true;
    for (const auto& ex : dataset) {
        exact = exact && result.cascade.evaluate(ex.inputs).to_string() == ex.target.to_string();
    }
    check(exact, "local greedy synthesis reproduces xor truth table");
    check(diagnostics.strategy == sle::SearchStrategy::LocalGreedy, "local greedy reports local strategy diagnostics");
}

void test_local_synthesis_xor() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x00});

    sle::SynthesisConfig cfg;
    cfg.iterations = 2048;
    auto result = sle::synthesize_local(cascade, dataset, cfg);
    bool exact = true;
    for (const auto& ex : dataset) {
        exact = exact && result.cascade.evaluate(ex.inputs).to_string() == ex.target.to_string();
    }
    check(exact, "local synthesis reproduces xor truth table");
    check(result.fitness.output_fitness >= 0.999999, "local synthesis reaches perfect output fitness");
}

void test_residual_synthesis_xor3_from_or_base() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(true)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(true)},
        {{bit(true), bit(true), bit(false)}, bit(true)},
        {{bit(true), bit(true), bit(true)}, bit(false)},
    };

    sle::BooleanCascade base(3);
    base.add_gate({0, 1, 2, 0xFE});

    sle::SynthesisConfig cfg;
    cfg.iterations = 4096;
    auto result = sle::synthesize_with_residual(base, dataset, cfg);
    bool exact = true;
    for (const auto& ex : dataset) {
        auto out = sle::residual_apply(result.base.cascade.evaluate(ex.inputs), result.residual.cascade.evaluate(ex.inputs));
        exact = exact && out.to_string() == ex.target.to_string();
    }
    check(exact, "residual synthesis reproduces xor3 from or-base");
    check(result.final_fitness.output_fitness >= 0.999999, "residual synthesis reaches perfect final output");
}

void test_multigate_api() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
        {{bit(false), bit(false), bit(false)}, bit(false)},
    };

    sle::SynthesisConfig cfg;
    cfg.iterations = 256;
    auto result = sle::synthesize_multigate(3, 2, dataset, cfg);
    check(result.cascade.gate_count() == 2, "multigate synthesis creates requested gate count");
}

void test_full_engine_train_and_run() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::FullEngineConfig cfg;
    cfg.gate_count = 3;
    cfg.synthesis.iterations = 4096;
    cfg.solver_policy.residual_policy = sle::ResidualPolicyMode::FallbackOnly;

    sle::HardLogicContract contract;
    contract.required = sle::BitVector(1, false);
    contract.forbidden = sle::BitVector(1, false);

    auto trained = sle::train_full_engine(dataset, cfg, contract);
    check(trained.fitness.output_fitness >= 0.999999, "full engine reaches perfect output fitness on xor3 dataset");
    check(trained.selected_tier == sle::SynthesisTier::Exact
            || trained.selected_tier == sle::SynthesisTier::Local
            || trained.selected_tier == sle::SynthesisTier::MonteCarloTreeSearch
            || trained.selected_tier == sle::SynthesisTier::Residual,
          "full engine reports selected synthesis tier");
    check(!trained.tier_diagnostics.empty(), "full engine exposes tier diagnostics");
    check(!trained.selection_summary.empty(), "full engine exposes selection summary");

    for (const auto& ex : dataset) {
        auto out = sle::run_full_engine(trained.model, ex.inputs);
        check(out.to_string() == ex.target.to_string(), "full engine inference matches target");
    }
}

void test_bitstream_dataset_builder() {
    sle::ScalarBitstreamConfig cfg;
    cfg.stream_length = 64;
    auto dataset = sle::make_bitstream_dataset({{0.1, 0.9}, {0.8, 0.2}}, {0.0, 1.0}, cfg);
    check(dataset.kind() == sle::TaskKind::StochasticBitstream, "bitstream dataset uses stochastic task kind");
    check(dataset.size() == 2, "bitstream dataset preserves sample count");
    check(dataset.samples().front().inputs.front().size() == 64, "bitstream dataset uses configured stream length");
}

void test_framework_api() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };
    std::vector<sle::TrainingExample> examples{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    auto dataset = sle::make_boolean_dataset(examples);
    sle::TrainerConfig cfg;
    cfg.engine.gate_count = 4;
    cfg.engine.synthesis.iterations = 4096;
    cfg.engine.solver_policy.residual_policy = sle::ResidualPolicyMode::FallbackOnly;
    sle::Trainer trainer(cfg);
    sle::HardLogicContract contract;
    contract.required = sle::BitVector(1, false);
    contract.forbidden = sle::BitVector(1, false);
    auto model = trainer.fit(dataset, contract);
    auto metrics = trainer.evaluate(model, dataset, contract);
    check(metrics.errors == 0, "framework trainer evaluates without errors on training set");
    check(metrics.output_fitness >= 0.999999, "framework metrics report perfect fitness");
}

void test_solver_policy_and_exact_pattern_layer() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::BooleanCascade initial(3);
    initial.add_gate({0, 1, 2, 0x00});

    auto pattern = sle::detect_exact_pattern(initial, dataset, 3);
    check(pattern.has_value(), "exact pattern layer recognizes supported pattern");

    auto registry = sle::exact_pattern_registry(initial, dataset, 3);
    check(!registry.empty(), "exact pattern registry returns candidates");
    check(!registry.front().name.empty(), "exact pattern registry provides pattern names");

    sle::SolverPolicy policy;
    policy.tiers = {
        {sle::SynthesisTier::Exact, true, 0, true},
        {sle::SynthesisTier::MonteCarloTreeSearch, true, 32, false},
    };
    policy.residual_policy = sle::ResidualPolicyMode::Disabled;

    std::vector<sle::TierDiagnostics> diagnostics;
    auto candidates = sle::synthesize_tiered(initial, dataset, sle::SynthesisConfig{}, policy, &diagnostics);
    check(!candidates.empty(), "solver policy runs configured tiers");
    check(!diagnostics.empty(), "tier policy emits diagnostics");
    check(!diagnostics.front().tier_name.empty(), "tier diagnostics provide string tier names");
}

void test_task_kind_policy_and_residual_planner() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    auto boolean_cfg = sle::default_trainer_config_for_task(sle::TaskKind::BooleanFunction);
    auto stochastic_cfg = sle::default_trainer_config_for_task(sle::TaskKind::StochasticBitstream);
    check(!boolean_cfg.engine.solver_policy.tiers.empty(), "default boolean task policy is defined");
    check(!stochastic_cfg.engine.solver_policy.tiers.empty(), "default stochastic task policy is defined");

    std::vector<sle::TrainingExample> dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(true)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(true)},
        {{bit(true), bit(true), bit(false)}, bit(true)},
        {{bit(true), bit(true), bit(true)}, bit(false)},
    };

    sle::BooleanCascade base(3);
    base.add_gate({0, 1, 2, 0xFE});

    sle::ResidualPlannerConfig planner;
    planner.tiers = {
        {sle::SynthesisTier::Exact, true, 0, false},
        {sle::SynthesisTier::Local, true, 0, false},
    };
    auto residual = sle::synthesize_with_residual(base, dataset, sle::SynthesisConfig{}, planner);
    check(!residual.steps.empty(), "residual planner records executed steps");
    check(!residual.summary.empty(), "residual planner exposes summary");
}

void test_baked_inference_rehydration_and_policy() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x96});

    sle::BakeConfig bake;
    bake.jit.word_count = 1;
    bake.jit.preferred_backend = sle::JitBackend::ScalarGpr;
    bake.inference_policy.mode = sle::ExecutionMode::InferenceOnly;

    auto artifact = sle::bake_inference_artifact(cascade, bake);
    check(artifact.prepared, "baked inference artifact is prepared after bake");
    check(artifact.target_instruction_set == sle::TargetInstructionSet::ScalarX86_64,
          "baked inference artifact stores target instruction set");
    check(!artifact.code_bytes.empty(), "baked inference artifact stores raw jit bytes");

    artifact.compiled.reset();
    artifact.prepared = false;
    sle::prepare_inference_runtime(artifact);
    check(artifact.prepared && artifact.compiled != nullptr, "baked inference artifact rehydrates compiled kernel");

    auto result = sle::run_baked_inference(artifact, {bit(true), bit(false), bit(true)}, bake.inference_policy);
    check(result.ok, "baked inference executes in inference-only mode");
    check(result.output.to_string() == bit(false).to_string(), "baked inference output matches xor");

    auto bad = sle::run_baked_inference(artifact, {bit(true), bit(false)}, bake.inference_policy);
    check(!bad.ok && bad.error == sle::InferenceError::InputShapeMismatch,
          "baked inference rejects mismatched input shape");
}

void test_baked_inference_serialization_and_resolver() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    std::vector<sle::TrainingExample> xor_dataset{
        {{bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true)}, bit(false)},
        {{bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true)}, bit(false)},
        {{bit(true), bit(true), bit(false)}, bit(false)},
        {{bit(true), bit(true), bit(true)}, bit(true)},
    };

    sle::BooleanCascade initial(3);
    initial.add_gate({0, 1, 2, 0x00});

    sle::BakeConfig bake;
    bake.jit.word_count = 1;
    bake.jit.preferred_backend = sle::JitBackend::ScalarGpr;
    bake.inference_policy.mode = sle::ExecutionMode::InferenceOnly;

    sle::CompilationCache cache(bake.jit, 8);
    auto resolved = sle::resolve_inference_only_artifact(initial, xor_dataset, bake, &cache);
    check(resolved.found, "inference-only resolver finds exact-registry artifact");

    const auto tmp = std::filesystem::temp_directory_path() / "sle_baked_artifact.bin";
    sle::save_baked_inference_artifact(resolved.artifact, tmp.string());
    auto loaded = sle::load_baked_inference_artifact(tmp.string());
    check(!loaded.prepared, "loaded baked artifact requires explicit rehydration");
    sle::prepare_inference_runtime(loaded);
    auto run = sle::run_baked_inference(loaded, {bit(true), bit(false), bit(true)}, bake.inference_policy);
    check(run.ok, "loaded baked artifact runs after rehydration");
    check(run.output.to_string() == bit(false).to_string(), "loaded baked artifact preserves inference output");
    std::filesystem::remove(tmp);

    std::vector<sle::TrainingExample> unsupported_dataset{
        {{bit(false), bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(false), bit(true)}, bit(false)},
        {{bit(false), bit(false), bit(true), bit(false)}, bit(false)},
        {{bit(false), bit(false), bit(true), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(false), bit(false)}, bit(false)},
        {{bit(false), bit(true), bit(false), bit(true)}, bit(true)},
        {{bit(false), bit(true), bit(true), bit(false)}, bit(true)},
        {{bit(false), bit(true), bit(true), bit(true)}, bit(true)},
        {{bit(true), bit(false), bit(false), bit(false)}, bit(false)},
        {{bit(true), bit(false), bit(false), bit(true)}, bit(true)},
        {{bit(true), bit(false), bit(true), bit(false)}, bit(true)},
        {{bit(true), bit(false), bit(true), bit(true)}, bit(true)},
        {{bit(true), bit(true), bit(false), bit(false)}, bit(true)},
        {{bit(true), bit(true), bit(false), bit(true)}, bit(true)},
        {{bit(true), bit(true), bit(true), bit(false)}, bit(true)},
        {{bit(true), bit(true), bit(true), bit(true)}, bit(false)},
    };
    sle::BooleanCascade unsupported_initial(4);
    unsupported_initial.add_gate({0, 1, 2, 0x00});
    auto miss = sle::resolve_inference_only_artifact(unsupported_initial, unsupported_dataset, bake, nullptr);
    check(!miss.found && miss.error == sle::InferenceError::PatternNotFound,
          "inference-only resolver returns explicit pattern-not-found");
}

void test_isa_mismatch_and_branchless_artifact_scaffold() {
    auto bit = [](bool v) {
        sle::BitVector b(1);
        b.set(0, v);
        return b;
    };

    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x96});

    sle::BakeConfig bake;
    bake.jit.word_count = 1;
    bake.jit.preferred_backend = sle::JitBackend::ScalarGpr;
    auto artifact = sle::bake_inference_artifact(cascade, bake);
    artifact.compiled.reset();
    artifact.prepared = false;
    artifact.target_instruction_set = static_cast<sle::TargetInstructionSet>(999);
    bool threw = false;
    try {
        sle::prepare_inference_runtime(artifact);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "baked artifact preparation hard-fails on ISA mismatch");

    sle::BranchlessDecisionArtifactSpec spec;
    spec.predicate_count = 2;
    spec.leaf_count = 2;
    spec.program = {0x01, 0x02, 0x03};
    auto branchless = sle::bake_branchless_decision_artifact(spec, bake);
    check(branchless.artifact_kind == sle::BakedArtifactKind::BranchlessDecisionLogic,
          "branchless decision artifact scaffold uses dedicated artifact kind");
    auto branchless_run = sle::run_baked_inference(branchless, {bit(true), bit(false)}, sle::InferencePolicy{sle::ExecutionMode::InferenceOnly, true, true, false});
    check(branchless_run.ok, "branchless decision artifact scaffold participates in inference-only path");
}

} // namespace

int main() {
    try {
        test_ternary_xor();
        test_mixer_invertibility();
        test_residual_exactness();
        test_forbidden_filter();
        test_required_verification();
        test_engine_path();
        test_compiled_cascade_matches_interpreter();
        test_jit_patch_in_place_binary();
        test_jit_first_trainer_finds_xor();
        test_prepare_batch_transposes_bitplanes();
        test_jit_trainer_selection_and_expansion();
        test_mdl_prefers_shorter_program();
        test_fast_matches_scalar();
        test_mcts_synthesis_xor();
        test_local_greedy_synthesis_xor();
        test_local_synthesis_xor();
        test_residual_synthesis_xor3_from_or_base();
        test_multigate_api();
        test_full_engine_train_and_run();
        test_bitstream_dataset_builder();
        test_framework_api();
        test_solver_policy_and_exact_pattern_layer();
        test_task_kind_policy_and_residual_planner();
        test_baked_inference_rehydration_and_policy();
        test_baked_inference_serialization_and_resolver();
        test_isa_mismatch_and_branchless_artifact_scaffold();
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << '\n';
        return 2;
    }

    std::cout << "Passed: " << g_pass << ", Failed: " << g_fail << '\n';
    return g_fail == 0 ? 0 : 1;
}
