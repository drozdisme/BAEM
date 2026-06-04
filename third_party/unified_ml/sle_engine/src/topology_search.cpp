#include "sle/topology_search.hpp"
#include "sle/jit_batch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace sle {
namespace {

struct ScoredAction {
    MutationAction action;
    FitnessBreakdown fitness;
    double heuristic = 0.0;
    bool cache_hit = false;
    double structural_cost = 1.0;
};

struct TreeNode {
    explicit TreeNode(BooleanCascade state_, FitnessBreakdown fitness_, std::size_t parent_ = npos,
                      MutationAction incoming_ = {})
        : state(std::move(state_)), fitness(fitness_), parent(parent_), incoming_action(incoming_) {}

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    BooleanCascade state;
    FitnessBreakdown fitness;
    std::size_t parent = npos;
    MutationAction incoming_action{};
    std::vector<std::size_t> children;
    std::vector<ScoredAction> pending_actions;
    std::size_t visits = 0;
    double value_sum = 0.0;
    std::size_t depth = 0;
};

double normalized_gain(double candidate, double baseline) {
    return 0.5 + 0.5 * std::tanh((candidate - baseline) * 6.0);
}

double compute_progress_ratio(const FitnessBreakdown& root, const FitnessBreakdown& best) {
    const double gap = std::max(1e-6, 1.0 - root.output_fitness);
    const double improvement = std::max(0.0, best.output_fitness - root.output_fitness);
    return std::clamp(improvement / gap, 0.0, 1.0);
}

double dynamic_exploration_constant(std::size_t iteration,
                                    std::size_t total_iterations,
                                    const FitnessBreakdown& root_fitness,
                                    const FitnessBreakdown& best_fitness,
                                    const SynthesisConfig& config) {
    const double epoch = static_cast<double>(iteration + 1) / static_cast<double>(std::max<std::size_t>(1, total_iterations));
    const double progress = compute_progress_ratio(root_fitness, best_fitness);
    const double mdl_progress = std::clamp((root_fitness.mdl_model_bits - best_fitness.mdl_model_bits)
        / std::max(1.0, root_fitness.mdl_model_bits + root_fitness.mdl_residual_bits), -1.0, 1.0);
    const double scale = 1.0 + config.uct_epoch_decay * (1.0 - epoch)
        + config.uct_mdl_sensitivity * (1.0 - progress)
        + 0.25 * std::max(0.0, mdl_progress);
    return config.uct_exploration_base * scale;
}

double action_structural_cost(const MutationAction& action, bool cache_hit) {
    if (action.kind != MutationKind::MacroTemplate) return 1.0;
    return cache_hit ? 2.0 : 6.0;
}

std::size_t safe_input_ref(const BooleanCascade& cascade, std::size_t preferred) {
    if (cascade.input_count() == 0) return 0;
    return std::min(preferred, cascade.input_count() - 1);
}

bool macro_action_is_valid(const BooleanCascade& cascade, const MutationAction& action) {
    if (action.kind != MutationKind::MacroTemplate) return true;
    if (cascade.input_count() == 0) return false;
    const std::size_t max_gate_count = std::max<std::size_t>(cascade.gate_count(), 1);
    if (action.gate_index > max_gate_count) return false;
    switch (action.macro_kind) {
        case MacroTemplateKind::ThreeInputXor:
        case MacroTemplateKind::Multiplexer:
        case MacroTemplateKind::TwoStageXor:
            return cascade.input_count() >= 3;
        case MacroTemplateKind::DelayCascade:
            return true;
        case MacroTemplateKind::None:
            return false;
    }
    return false;
}

double score_action_prior(const MutationAction& action, const PerceptionSummary* perception) {
    if (perception == nullptr) return 0.0;
    if (perception->channels.empty()) return 0.0;

    auto channel_score = [&](std::size_t source) {
        if (source >= perception->channels.size()) return 0.25 * perception->mean_abs_correlation;
        const auto& stats = perception->channels[source];
        return 0.65 * stats.target_correlation + 0.20 * stats.entropy + 0.15 * (1.0 - std::abs(0.5 - stats.one_ratio) * 2.0);
    };

    switch (action.kind) {
        case MutationKind::MacroTemplate:
            switch (action.macro_kind) {
                case MacroTemplateKind::ThreeInputXor:
                    return 0.9 + perception->mean_abs_correlation;
                case MacroTemplateKind::Multiplexer:
                    return 0.75 + 0.5 * perception->target_entropy;
                case MacroTemplateKind::DelayCascade:
                    return 0.55 + 0.3 * perception->target_entropy;
                case MacroTemplateKind::TwoStageXor:
                    return 0.8 + 0.7 * perception->mean_abs_correlation;
                case MacroTemplateKind::None:
                    return 0.0;
            }
            break;
        case MutationKind::RewireInputA:
        case MutationKind::RewireInputB:
        case MutationKind::RewireInputC:
            return channel_score(action.new_source);
        case MutationKind::FlipMaskBit:
            return 0.15;
    }
    return 0.0;
}

FitnessBreakdown evaluate_cached_fitness(const BooleanCascade& cascade,
                                         const std::vector<TrainingExample>& dataset,
                                         const SynthesisConfig& config,
                                         const SafeBatchedJITEvaluator* jit_evaluator,
                                         CompilationCache* cache,
                                         bool* cache_hit_out = nullptr) {
    bool cache_hit = false;
    auto fitness = evaluate_fitness(cascade, dataset, config);
    if (cache != nullptr && jit_evaluator != nullptr) {
        const auto signature = topology_signature(cascade);
        cache_hit = cache->contains(signature);
        try {
            auto cached = cache_hit ? cache->get(signature) : cache->get_or_compile(cascade);
            if (cached.compiled) {
                if (cached.compiled->backend() == JitBackend::Avx512Ternlog) {
                    cached.compiled->patch_all_masks(cascade);
                }
                fitness = jit_evaluator->evaluate(*cached.compiled, cascade, config);
                cache_hit = true;
            }
        } catch (...) {
        }
    }
    if (cache_hit_out != nullptr) *cache_hit_out = cache_hit;
    return fitness;
}

std::vector<ScoredAction> rank_actions(const BooleanCascade& cascade,
                                       const std::vector<TrainingExample>& dataset,
                                       const SynthesisConfig& config,
                                       const PerceptionSummary* perception,
                                       CompilationCache* cache) {
    const auto baseline = evaluate_fitness(cascade, dataset, config);
    std::vector<ScoredAction> ranked;
    SafeBatchedJITEvaluator jit_evaluator;
    bool use_jit_cache = false;
    if (cache != nullptr) {
        const bool supports_cached_jit = std::all_of(cascade.gates().begin(), cascade.gates().end(), [](const TernaryGate& gate) {
            return gate.mask == 0x00 || gate.mask == 0xFF;
        });
        if (supports_cached_jit) {
            try {
                jit_evaluator = make_safe_batched_jit_evaluator(dataset, 64, 64);
                use_jit_cache = true;
            } catch (...) {
            }
        }
    }

    for (const auto& action : enumerate_mutation_actions(cascade)) {
        if (!macro_action_is_valid(cascade, action)) continue;
        auto candidate = apply_mutation_action(cascade, action);
        bool cache_hit = false;
        auto fitness = evaluate_cached_fitness(candidate,
                                               dataset,
                                               config,
                                               use_jit_cache ? &jit_evaluator : nullptr,
                                               cache,
                                               &cache_hit);
        const double prior = perception == nullptr ? 0.0 : config.perception_prior_weight * score_topology_prior(candidate, *perception);
        const double structural_cost = action_structural_cost(action, cache_hit);
        const double heuristic = normalized_gain(fitness.total, baseline.total)
            + score_action_prior(action, perception)
            + prior
            - 0.03 * structural_cost;
        ranked.push_back(ScoredAction{action, fitness, heuristic, cache_hit, structural_cost});
    }

    std::sort(ranked.begin(), ranked.end(), [](const ScoredAction& lhs, const ScoredAction& rhs) {
        if (lhs.heuristic == rhs.heuristic) return lhs.fitness.total > rhs.fitness.total;
        return lhs.heuristic > rhs.heuristic;
    });
    return ranked;
}

std::vector<ScoredAction> shortlist_actions(const std::vector<ScoredAction>& ranked,
                                            std::size_t limit,
                                            double baseline_total) {
    std::vector<ScoredAction> shortlisted;
    shortlisted.reserve(std::min(limit, ranked.size()));

    for (const auto& scored : ranked) {
        if (shortlisted.size() >= limit) break;
        if (scored.fitness.total + 1e-12 < baseline_total) continue;
        shortlisted.push_back(scored);
    }

    if (shortlisted.empty()) {
        for (const auto& scored : ranked) {
            if (shortlisted.size() >= limit) break;
            shortlisted.push_back(scored);
        }
    }

    return shortlisted;
}

std::size_t select_promising(const std::vector<TreeNode>& nodes,
                             std::size_t root_index,
                             std::size_t iteration,
                             std::size_t total_iterations,
                             const FitnessBreakdown& root_fitness,
                             const FitnessBreakdown& best_fitness,
                             const SynthesisConfig& config,
                             const PerceptionSummary* perception) {
    constexpr std::size_t kMaxDepth = 5;

    std::size_t current = root_index;
    while (!nodes[current].children.empty() && nodes[current].depth < kMaxDepth) {
        const auto& parent = nodes[current];
        const double log_parent = std::log(static_cast<double>(std::max<std::size_t>(1, parent.visits)) + 1.0);
        const double exploration_constant = dynamic_exploration_constant(iteration, total_iterations, root_fitness, best_fitness, config);
        auto best = std::max_element(parent.children.begin(), parent.children.end(), [&](std::size_t lhs_index, std::size_t rhs_index) {
            const auto& lhs = nodes[lhs_index];
            const auto& rhs = nodes[rhs_index];
            const double lhs_exploit = lhs.visits == 0 ? 0.0 : lhs.value_sum / static_cast<double>(lhs.visits);
            const double rhs_exploit = rhs.visits == 0 ? 0.0 : rhs.value_sum / static_cast<double>(rhs.visits);
            const double lhs_explore = lhs.visits == 0
                ? std::numeric_limits<double>::infinity()
                : exploration_constant * std::sqrt(log_parent / static_cast<double>(lhs.visits));
            const double rhs_explore = rhs.visits == 0
                ? std::numeric_limits<double>::infinity()
                : exploration_constant * std::sqrt(log_parent / static_cast<double>(rhs.visits));
            const double lhs_prior = 0.1 * score_action_prior(lhs.incoming_action, perception);
            const double rhs_prior = 0.1 * score_action_prior(rhs.incoming_action, perception);
            return (lhs_exploit + lhs_explore + lhs_prior) < (rhs_exploit + rhs_explore + rhs_prior);
        });
        current = *best;
    }
    return current;
}

std::size_t expand_node(std::vector<TreeNode>& nodes,
                        std::size_t node_index,
                        const std::vector<TrainingExample>& dataset,
                        const SynthesisConfig& config,
                        const PerceptionSummary* perception,
                        CompilationCache* cache,
                        std::size_t& expanded_nodes) {
    constexpr std::size_t kMaxDepth = 5;
    constexpr std::size_t kExpansionWidth = 40;

    auto& node = nodes[node_index];
    if (node.depth >= kMaxDepth) return node_index;
    if (node.pending_actions.empty()) {
        auto ranked = rank_actions(node.state, dataset, config, perception, cache);
        node.pending_actions = shortlist_actions(ranked, kExpansionWidth, node.fitness.total);
    }
    if (node.pending_actions.empty()) return node_index;

    const auto scored = node.pending_actions.front();
    node.pending_actions.erase(node.pending_actions.begin());

    auto candidate = apply_mutation_action(node.state, scored.action);
    TreeNode child(std::move(candidate), scored.fitness, node_index, scored.action);
    child.depth = node.depth + 1;
    nodes.push_back(std::move(child));
    const std::size_t child_index = nodes.size() - 1;
    nodes[node_index].children.push_back(child_index);
    ++expanded_nodes;
    return child_index;
}

double rollout_from(const BooleanCascade& seed_state,
                    const FitnessBreakdown& seed_fitness,
                    std::mt19937_64& rng,
                    const std::vector<TrainingExample>& dataset,
                    const SynthesisConfig& config,
                    const PerceptionSummary* perception,
                    CompilationCache* cache,
                    std::size_t& rollout_evaluations,
                    BooleanCascade& best_state,
                    FitnessBreakdown& best_fitness,
                    std::size_t& accepted) {
    auto current = seed_state;
    auto current_fitness = seed_fitness;
    double best_reward = normalized_gain(current_fitness.total, seed_fitness.total);
    std::size_t stale_steps = 0;

    SafeBatchedJITEvaluator jit_evaluator;
    bool use_jit_cache = false;
    if (cache != nullptr) {
        const bool supports_cached_jit = std::all_of(current.gates().begin(), current.gates().end(), [](const TernaryGate& gate) {
            return gate.mask == 0x00 || gate.mask == 0xFF;
        });
        if (supports_cached_jit) {
            try {
                jit_evaluator = make_safe_batched_jit_evaluator(dataset, 64, 64);
                use_jit_cache = true;
            } catch (...) {
            }
        }
    }

    const std::size_t budget = std::clamp<std::size_t>(config.rollout_budget, 1, 100);
    double spent_budget = 0.0;
    for (std::size_t step = 0; step < budget && current.gate_count() > 0 && spent_budget < static_cast<double>(budget); ++step) {
        auto actions = enumerate_mutation_actions(current);
        actions.erase(std::remove_if(actions.begin(), actions.end(), [&](const auto& action) {
            return !macro_action_is_valid(current, action);
        }), actions.end());
        if (actions.empty()) break;

        MutationAction selected_action = actions.front();
        if (perception != nullptr) {
            double best_action_score = -std::numeric_limits<double>::infinity();
            std::uniform_int_distribution<std::size_t> sample_dist(0, actions.size() - 1);
            const std::size_t sample_budget = std::min<std::size_t>(12, actions.size());
            for (std::size_t sample = 0; sample < sample_budget; ++sample) {
                const auto& candidate_action = actions[sample_dist(rng)];
                const double score = score_action_prior(candidate_action, perception);
                if (score > best_action_score) {
                    best_action_score = score;
                    selected_action = candidate_action;
                }
            }
        } else {
            std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
            selected_action = actions[dist(rng)];
        }

        auto candidate = apply_mutation_action(current, selected_action);
        bool cache_hit = false;
        auto candidate_fitness = evaluate_cached_fitness(candidate,
                                                         dataset,
                                                         config,
                                                         use_jit_cache ? &jit_evaluator : nullptr,
                                                         cache,
                                                         &cache_hit);
        spent_budget += action_structural_cost(selected_action, cache_hit);
        ++rollout_evaluations;

        const double reward = normalized_gain(candidate_fitness.total, seed_fitness.total);
        best_reward = std::max(best_reward, reward);

        if (candidate_fitness.total >= current_fitness.total) {
            current = candidate;
            current_fitness = candidate_fitness;
            ++accepted;
            stale_steps = 0;
        } else {
            ++stale_steps;
        }

        if (candidate_fitness.total > best_fitness.total) {
            best_state = candidate;
            best_fitness = candidate_fitness;
        }
        if (stale_steps >= config.rollout_patience) break;
    }

    return best_reward;
}

void backpropagate(std::vector<TreeNode>& nodes, std::size_t node_index, double reward) {
    while (node_index != TreeNode::npos) {
        auto& node = nodes[node_index];
        ++node.visits;
        node.value_sum += reward;
        node_index = node.parent;
    }
}

void build_principal_path(const std::vector<TreeNode>& nodes,
                          std::size_t root_index,
                          SearchDiagnostics& diagnostics) {
    diagnostics.principal_path.clear();
    std::size_t current = root_index;
    while (!nodes[current].children.empty()) {
        auto best = std::max_element(nodes[current].children.begin(), nodes[current].children.end(), [&](std::size_t lhs_index, std::size_t rhs_index) {
            return nodes[lhs_index].visits < nodes[rhs_index].visits;
        });
        current = *best;
        const auto& node = nodes[current];
        diagnostics.principal_path.push_back(SearchTraceEntry{
            node.depth,
            node.visits,
            node.visits == 0 ? 0.0 : node.value_sum / static_cast<double>(node.visits),
            node.fitness,
            node.incoming_action,
        });
    }
}

void ensure_gate_index(BooleanCascade& cascade, std::size_t gate_index) {
    while (cascade.gate_count() <= gate_index) {
        const std::size_t available = cascade.input_count() + cascade.gate_count();
        const std::size_t a = available > 0 ? 0 : 0;
        const std::size_t b = available > 1 ? 1 : a;
        const std::size_t c = available > 2 ? 2 : b;
        cascade.add_gate({a, b, c, 0x00});
    }
}

void set_gate(BooleanCascade& cascade, std::size_t gate_index, const TernaryGate& gate) {
    ensure_gate_index(cascade, gate_index);
    cascade.mutable_gates()[gate_index] = gate;
}

} // namespace

std::vector<MutationAction> enumerate_macro_actions(const BooleanCascade& cascade) {
    std::vector<MutationAction> actions;
    const std::size_t max_anchor = std::max<std::size_t>(cascade.gate_count(), 1);
    for (std::size_t gate_index = 0; gate_index < max_anchor; ++gate_index) {
        actions.push_back(MutationAction{MutationKind::MacroTemplate, gate_index, 0, 0, 0, MacroTemplateKind::ThreeInputXor, 1});
        actions.push_back(MutationAction{MutationKind::MacroTemplate, gate_index, 0, 0, 0, MacroTemplateKind::Multiplexer, 1});
        actions.push_back(MutationAction{MutationKind::MacroTemplate, gate_index, 0, 0, 0, MacroTemplateKind::DelayCascade, 2});
        actions.push_back(MutationAction{MutationKind::MacroTemplate, gate_index, 0, 0, 0, MacroTemplateKind::TwoStageXor, 2});
    }
    return actions;
}

std::vector<MutationAction> enumerate_mutation_actions(const BooleanCascade& cascade) {
    std::vector<MutationAction> actions;
    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        for (std::uint8_t bit = 0; bit < 8; ++bit) {
            actions.push_back(MutationAction{MutationKind::FlipMaskBit, gate_index, 0, 0, bit});
        }

        const std::size_t available_inputs = cascade.input_count() + gate_index;
        for (std::size_t source = 0; source < available_inputs; ++source) {
            actions.push_back(MutationAction{MutationKind::RewireInputA, gate_index, 0, source, 0});
            actions.push_back(MutationAction{MutationKind::RewireInputB, gate_index, 1, source, 0});
            actions.push_back(MutationAction{MutationKind::RewireInputC, gate_index, 2, source, 0});
        }
    }

    auto macros = enumerate_macro_actions(cascade);
    actions.insert(actions.end(), macros.begin(), macros.end());
    return actions;
}

BooleanCascade apply_mutation_action(const BooleanCascade& cascade, const MutationAction& action) {
    auto mutated = cascade;

    if (action.kind == MutationKind::MacroTemplate) {
        if (!macro_action_is_valid(cascade, action)) return mutated;
        const std::size_t base = std::min(action.gate_index, mutated.gate_count());
        const std::size_t in0 = safe_input_ref(mutated, 0);
        const std::size_t in1 = safe_input_ref(mutated, 1);
        const std::size_t in2 = safe_input_ref(mutated, 2);
        switch (action.macro_kind) {
            case MacroTemplateKind::ThreeInputXor:
                set_gate(mutated, base, {in0, in1, in2, 0x96});
                break;
            case MacroTemplateKind::Multiplexer:
                set_gate(mutated, base, {in0, in1, in2, 0xCA});
                break;
            case MacroTemplateKind::DelayCascade: {
                set_gate(mutated, base, {in0, in0, in0, 0xF0});
                const std::size_t stage0 = mutated.input_count() + base;
                set_gate(mutated, base + 1, {stage0, stage0, stage0, 0xF0});
                break;
            }
            case MacroTemplateKind::TwoStageXor: {
                set_gate(mutated, base, {in0, in1, in1, 0x66});
                const std::size_t intermediate = mutated.input_count() + base;
                set_gate(mutated, base + 1, {intermediate, in2, in2, 0x66});
                break;
            }
            case MacroTemplateKind::None:
                break;
        }
        return mutated;
    }

    if (action.gate_index >= mutated.gate_count()) return mutated;

    auto& gate = mutated.mutable_gates()[action.gate_index];
    switch (action.kind) {
        case MutationKind::FlipMaskBit:
            gate.mask = static_cast<std::uint8_t>(gate.mask ^ static_cast<std::uint8_t>(1U << action.bit_index));
            break;
        case MutationKind::RewireInputA:
            gate.a = action.new_source;
            break;
        case MutationKind::RewireInputB:
            gate.b = action.new_source;
            break;
        case MutationKind::RewireInputC:
            gate.c = action.new_source;
            break;
        case MutationKind::MacroTemplate:
            break;
    }
    return mutated;
}

std::string describe_mutation_action(const MutationAction& action) {
    std::ostringstream oss;
    switch (action.kind) {
        case MutationKind::FlipMaskBit:
            oss << "flip-mask(g=" << action.gate_index << ", bit=" << static_cast<int>(action.bit_index) << ")";
            break;
        case MutationKind::RewireInputA:
        case MutationKind::RewireInputB:
        case MutationKind::RewireInputC:
            oss << "rewire(g=" << action.gate_index << ", src=" << action.new_source << ")";
            break;
        case MutationKind::MacroTemplate:
            oss << "macro(g=" << action.gate_index << ", kind=" << static_cast<int>(action.macro_kind)
                << ", span=" << action.macro_span << ")";
            break;
    }
    return oss.str();
}

SynthesisResult synthesize_mcts(BooleanCascade initial,
                                const std::vector<TrainingExample>& dataset,
                                const SynthesisConfig& config,
                                SearchDiagnostics* diagnostics,
                                const PerceptionSummary* perception,
                                CompilationCache* cache) {
    if (dataset.empty()) throw std::invalid_argument("dataset must not be empty");
    if (initial.gate_count() == 0) throw std::invalid_argument("initial cascade must contain at least one gate");

    PerceptionSummary inferred_perception;
    if (perception == nullptr) {
        inferred_perception = summarize_bitstreams(dataset);
        perception = &inferred_perception;
    }

    std::mt19937_64 rng(config.seed);
    const auto root_fitness = evaluate_fitness(initial, dataset, config);

    std::vector<TreeNode> nodes;
    nodes.reserve(std::max<std::size_t>(64, std::min<std::size_t>(config.iterations + 1, 4096)));
    nodes.emplace_back(initial, root_fitness, TreeNode::npos);
    const std::size_t root_index = 0;

    auto best_state = initial;
    auto best_fitness = root_fitness;
    std::size_t accepted = 0;
    std::size_t expanded_nodes = 0;
    std::size_t rollout_evaluations = 0;

    for (std::size_t iter = 0; iter < config.iterations; ++iter) {
        const std::size_t selected_index = select_promising(nodes, root_index, iter, config.iterations, root_fitness, best_fitness, config, perception);
        const std::size_t expanded_index = expand_node(nodes, selected_index, dataset, config, perception, cache, expanded_nodes);
        const auto& expanded = nodes[expanded_index];

        const double reward = rollout_from(expanded.state,
                                           expanded.fitness,
                                           rng,
                                           dataset,
                                           config,
                                           perception,
                                           cache,
                                           rollout_evaluations,
                                           best_state,
                                           best_fitness,
                                           accepted);
        backpropagate(nodes, expanded_index, reward);

        if (expanded.fitness.total > best_fitness.total) {
            best_state = expanded.state;
            best_fitness = expanded.fitness;
            ++accepted;
        }

        if (expanded.fitness.total >= best_fitness.total) {
            nodes[expanded_index].pending_actions = rank_actions(nodes[expanded_index].state, dataset, config, perception, cache);
            if (!nodes[expanded_index].pending_actions.empty()) {
                const auto& top = nodes[expanded_index].pending_actions.front();
                if (top.fitness.total > best_fitness.total) {
                    best_state = apply_mutation_action(nodes[expanded_index].state, top.action);
                    best_fitness = top.fitness;
                    ++accepted;
                }
            }
        }
        if (best_fitness.output_fitness >= config.target_output_fitness) break;
    }

    SearchDiagnostics local_diagnostics;
    local_diagnostics.strategy = SearchStrategy::MonteCarloTreeSearch;
    local_diagnostics.explored_nodes = nodes[root_index].visits;
    local_diagnostics.expanded_nodes = expanded_nodes;
    local_diagnostics.rollout_evaluations = rollout_evaluations;
    if (cache != nullptr) {
        local_diagnostics.cache_hits = cache->hits();
        local_diagnostics.cache_misses = cache->misses();
    }
    build_principal_path(nodes, root_index, local_diagnostics);

    if (diagnostics != nullptr) *diagnostics = std::move(local_diagnostics);
    return SynthesisResult{best_state, best_fitness, accepted};
}

SynthesisResult synthesize_local_greedy(BooleanCascade initial,
                                        const std::vector<TrainingExample>& dataset,
                                        const SynthesisConfig& config,
                                        SearchDiagnostics* diagnostics,
                                        const PerceptionSummary* perception,
                                        CompilationCache* cache) {
    if (dataset.empty()) throw std::invalid_argument("dataset must not be empty");
    if (initial.gate_count() == 0) throw std::invalid_argument("initial cascade must contain at least one gate");

    PerceptionSummary inferred_perception;
    if (perception == nullptr) {
        inferred_perception = summarize_bitstreams(dataset);
        perception = &inferred_perception;
    }

    auto best_state = initial;
    auto best_fitness = evaluate_fitness(initial, dataset, config);
    std::size_t accepted = 0;
    std::size_t explored = 0;

    for (std::size_t iter = 0; iter < config.iterations; ++iter) {
        auto ranked = rank_actions(best_state, dataset, config, perception, cache);
        ++explored;
        if (ranked.empty()) break;

        const auto& top = ranked.front();
        if (top.fitness.total + 1e-12 < best_fitness.total) break;

        best_state = apply_mutation_action(best_state, top.action);
        best_fitness = top.fitness;
        ++accepted;

        if (best_fitness.output_fitness >= config.target_output_fitness) break;
    }

    if (diagnostics != nullptr) {
        diagnostics->strategy = SearchStrategy::LocalGreedy;
        diagnostics->explored_nodes = explored;
        diagnostics->expanded_nodes = accepted;
        diagnostics->rollout_evaluations = 0;
        if (cache != nullptr) {
            diagnostics->cache_hits = cache->hits();
            diagnostics->cache_misses = cache->misses();
        }
        diagnostics->principal_path.clear();
    }

    return SynthesisResult{best_state, best_fitness, accepted};
}

} // namespace sle
