#include "models/xgboost/tree/decision_tree.hpp"
#include "models/xgboost/tree/split_evaluator.hpp"
#include "models/xgboost/tree/leaf_wise_builder.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/utils/logger.hpp"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <cmath>
#include <numeric>
#include <random>

namespace xgb {

DecisionTree::DecisionTree(const TreeConfig& cfg)
    : cfg_(cfg)
{
    int max_nodes = (1 << (cfg_.max_depth + 1));
    nodes_.reserve(max_nodes);
}

//   Static scoring helpers                         
bst_float DecisionTree::calc_leaf_weight(
    double sum_grad, double sum_hess,
    bst_float lambda, bst_float alpha)
{
    double denom = sum_hess + static_cast<double>(lambda);
    if (std::fabs(denom) < 1e-9) return 0.f;
    if (alpha > 0.f) {
        double G = sum_grad, absG = std::fabs(G);
        double thresh = static_cast<double>(alpha);
        if (absG <= thresh) return 0.f;
        double shrunk = (G > 0.0 ? 1.0 : -1.0) * (absG - thresh);
        return static_cast<bst_float>(-shrunk / denom);
    }
    return static_cast<bst_float>(-sum_grad / denom);
}

bst_float DecisionTree::calc_gain(
    double sum_grad, double sum_hess, bst_float lambda)
{
    return static_cast<bst_float>(
        0.5 * (sum_grad * sum_grad) / (sum_hess + lambda));
}

bst_float DecisionTree::calc_split_gain(
    double GL, double HL, double GR, double HR,
    bst_float lambda, bst_float gamma)
{
    return static_cast<bst_float>(
        0.5 * ((GL * GL) / (HL + lambda)
              + (GR * GR) / (HR + lambda)
              - ((GL + GR) * (GL + GR)) / (HL + HR + lambda))
        - gamma);
}

NodeId DecisionTree::alloc_node(NodeId parent, bst_int depth) {
    NodeId id = static_cast<NodeId>(nodes_.size());
    TreeNode n;
    n.id = id; n.parent = parent; n.depth = depth;
    nodes_.push_back(n);
    return id;
}

void DecisionTree::make_leaf(NodeId nid, double sum_grad, double sum_hess) {
    auto& node   = nodes_[nid];
    node.is_leaf = true;
    node.leaf_value = calc_leaf_weight(sum_grad, sum_hess,
                                       cfg_.lambda, cfg_.alpha);
    node.sum_grad = sum_grad;
    node.sum_hess = sum_hess;
}

//                                        
//  Feature 2: colsample_bylevel  — per-level feature resampling
//
//  XGBoost supports three granularities of feature sampling:
//    • colsample_bytree  — one sample per tree (applied in Booster)
//    • colsample_bylevel — re-sample at each tree depth level  ← THIS
//    • colsample_bynode  — re-sample at every node (most expensive)
//
//  How it works:
//    At each call to build_node(), we check the current node's depth.
//    If colsample_bylevel < 1.0, we draw a random subset of size
//      k = max(1, round(n_cols * colsample_bylevel))
//    from the column indices provided by the caller.
//
//  Determinism: the RNG is seeded with  seed ^ depth  so that:
//    • The same depth always gets the same column subset for a given seed.
//    • Different depths get different subsets (depth is XOR'd in).
//    • Reproducibility is guaranteed by fixing BoosterConfig::seed.
//
//  No copies of feature data are created — we only resample the index vector.
//                                        
static std::vector<bst_uint> sample_cols_bylevel(
    const std::vector<bst_uint>& col_indices,
    bst_float colsample_bylevel,
    bst_uint seed,
    bst_int  depth)
{
    // No subsampling needed
    if (colsample_bylevel >= 1.0f ||
        col_indices.size() <= 1)
        return col_indices;

    const size_t n_total = col_indices.size();
    // k = max(1, floor(n * ratio))
    size_t k = std::max<size_t>(1,
        static_cast<size_t>(std::round(n_total * colsample_bylevel)));
    k = std::min(k, n_total);  // never exceed available columns

    // Deterministic RNG seeded by (global_seed XOR depth)
    // XOR with depth ensures each level draws a different subset.
    std::mt19937 rng(static_cast<uint32_t>(seed) ^
                     static_cast<uint32_t>(depth));

    // Partial Fisher-Yates shuffle: shuffle only the first k elements.
    // This is O(k), avoids copying the entire vector.
    std::vector<bst_uint> sampled = col_indices;  // one copy per level
    for (size_t i = 0; i < k; ++i) {
        // Uniform in [i, n_total - 1]
        std::uniform_int_distribution<size_t> dist(i, n_total - 1);
        size_t j = dist(rng);
        std::swap(sampled[i], sampled[j]);
    }
    sampled.resize(k);

    // Sort to preserve column-major access patterns (cache friendliness)
    std::sort(sampled.begin(), sampled.end());
    return sampled;
}

//   Public build entry point                        
void DecisionTree::build(
    const DMatrix& dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>& row_indices,
    const std::vector<bst_uint>& col_indices)
{
    nodes_.clear();
    if (row_indices.empty()) return;

    //   Feature 9: Leaf-wise (lossguide) growth               
    if (cfg_.grow_policy == "lossguide") {
        LeafWiseBuilder builder(nodes_, cfg_);
        builder.build(dm, grads, row_indices, col_indices);
        return;
    }

    //   Default: depth-wise growth                     
    if (cfg_.use_approx) {
        hist_builder_ = std::make_unique<HistogramBuilder>(cfg_);
        hist_builder_->build_cut_points(dm, grads, 256u);
    }

    alloc_node(kInvalidNodeId, 0);
    auto row_idx_copy = row_indices;
    build_node(kRootNodeId, dm, grads, row_idx_copy, col_indices, nullptr);
}

//                                        
//  build_node — Feature 2 (colsample_bylevel) + Feature 3 (hist subtraction)
//
//  colsample_bylevel is applied *before* the split search, replacing
//  col_indices with a level-specific subset. The subset is re-drawn each
//  time build_node is entered at a new depth. Children inherit the same
//  subset (the subset is fixed per call stack depth, not per node).
//                                        
void DecisionTree::build_node(
    NodeId nid,
    const DMatrix& dm,
    const std::vector<GradientPair>& grads,
    std::vector<bst_uint>& row_indices,
    const std::vector<bst_uint>& col_indices,
    std::vector<FeatureHistogram>* parent_hists)
{
    auto& node = nodes_[nid];

    //   Feature 2: resample columns at this depth level           
    // The resampled_cols vector is local to this invocation; children at
    // the same depth level will independently call sample_cols_bylevel with
    // the same seed^depth, producing the same subset — this matches
    // XGBoost's behaviour (bylevel = same set for all nodes at the same depth).
    // cfg_.seed is populated by GradientBooster::boost_one_round() with
    // (BoosterConfig::seed + tree_index) so each tree uses a unique seed
    // while being fully deterministic across runs.
    std::vector<bst_uint> level_cols = sample_cols_bylevel(
        col_indices,
        cfg_.colsample_bylevel,
        cfg_.seed,
        node.depth);

    // Aggregate stats
    double sum_grad = 0.0, sum_hess = 0.0;
    for (bst_uint i : row_indices) {
        sum_grad += grads[i].grad;
        sum_hess += grads[i].hess;
    }
    node.sum_grad  = sum_grad;
    node.sum_hess  = sum_hess;
    node.n_samples = static_cast<bst_uint>(row_indices.size());

    //   Stopping conditions                         
    if (node.depth >= cfg_.max_depth ||
        row_indices.size() <= 1 ||
        sum_hess < cfg_.min_child_weight)
    {
        make_leaf(nid, sum_grad, sum_hess);
        return;
    }

    //   Find best split (using level-resampled columns)           
    SplitEvaluator evaluator(cfg_);
    SplitCandidate best;

    if (cfg_.use_approx && hist_builder_ && parent_hists) {
        best = evaluator.find_best_split_approx_from_hists(
            *parent_hists, sum_grad, sum_hess, level_cols);
    } else {
        best = evaluator.find_best_split(dm, grads, row_indices, level_cols);
    }

    if (!best.valid || best.gain <= 0.f) {
        make_leaf(nid, sum_grad, sum_hess);
        return;
    }

    //   Apply split                             
    node.feature_idx  = best.feature_idx;
    node.split_value  = best.split_value;
    node.default_left = best.default_left;
    node.split_gain   = best.gain;
    node.is_leaf      = false;

    std::vector<bst_uint> left_rows, right_rows;
    left_rows.reserve(row_indices.size());
    right_rows.reserve(row_indices.size());
    for (bst_uint i : row_indices) {
        if (node.goes_left(dm.feature(i, best.feature_idx), false))
            left_rows.push_back(i);
        else
            right_rows.push_back(i);
    }

    if (left_rows.empty() || right_rows.empty()) {
        make_leaf(nid, sum_grad, sum_hess);
        return;
    }

    NodeId left_id  = alloc_node(nid, node.depth + 1);
    NodeId right_id = alloc_node(nid, node.depth + 1);
    nodes_[nid].child_left  = left_id;
    nodes_[nid].child_right = right_id;

    //   Feature 3: Histogram Subtraction                  
    if (cfg_.use_approx && hist_builder_) {
        bool left_is_larger = (left_rows.size() >= right_rows.size());
        const auto& large_rows = left_is_larger ? left_rows  : right_rows;
        const auto& small_rows = left_is_larger ? right_rows : left_rows;

        std::vector<FeatureHistogram> large_hists =
            hist_builder_->build_histograms(dm, grads, large_rows, level_cols);

        std::vector<FeatureHistogram> small_hists;
        if (parent_hists) {
            small_hists.resize(dm.num_features());
            for (bst_uint f : level_cols) {
                hist_builder_->subtract_histograms(
                    (*parent_hists)[f], large_hists[f], small_hists[f]);
            }
        } else {
            small_hists = hist_builder_->build_histograms(
                dm, grads, small_rows, level_cols);
        }

        NodeId large_id = left_is_larger ? left_id  : right_id;
        NodeId small_id = left_is_larger ? right_id : left_id;

        if (left_is_larger) {
            build_node(large_id, dm, grads, left_rows,  level_cols, &large_hists);
            build_node(small_id, dm, grads, right_rows, level_cols, &small_hists);
        } else {
            build_node(large_id, dm, grads, right_rows, level_cols, &large_hists);
            build_node(small_id, dm, grads, left_rows,  level_cols, &small_hists);
        }
    } else {
        // Note: children get level_cols, not col_indices — they will re-draw
        // their own subset when entering build_node at depth+1.
        build_node(left_id,  dm, grads, left_rows,  level_cols, nullptr);
        build_node(right_id, dm, grads, right_rows, level_cols, nullptr);
    }
}

//   Prediction                               
bst_float DecisionTree::predict(const DMatrix& dm, bst_uint row) const {
    if (nodes_.empty()) return 0.f;
    NodeId cur = kRootNodeId;
    while (true) {
        const auto& node = nodes_[cur];
        if (node.is_leaf) return node.leaf_value;
        bst_float val = dm.feature(row, node.feature_idx);
        cur = node.goes_left(val, false) ? node.child_left : node.child_right;
    }
}

std::vector<bst_float> DecisionTree::predict_batch(
    const DMatrix& dm,
    const std::vector<bst_uint>& row_indices) const
{
    std::vector<bst_float> out(row_indices.size());
    for (size_t i = 0; i < row_indices.size(); ++i)
        out[i] = predict(dm, row_indices[i]);
    return out;
}

// Multiply all leaf values by factor — called by GradientBooster to bake
// the per-round learning-rate (eta) into the tree so that predict() returns
// already-scaled values and global cfg_.eta need not be applied again.
void DecisionTree::scale_leaves_by(bst_float factor)
{
    for (auto& node : nodes_)
        if (node.is_leaf)
            node.leaf_value *= factor;
}

void DecisionTree::compute_feature_importance(
    std::vector<bst_float>& importance,
    bst_uint n_features) const
{
    if (importance.size() < static_cast<size_t>(n_features))
        importance.resize(n_features, 0.f);
    for (const auto& node : nodes_)
        if (!node.is_leaf && node.split_gain > 0.f)
            importance[node.feature_idx] += node.split_gain;
}

bst_int DecisionTree::num_leaves() const {
    bst_int c = 0;
    for (const auto& n : nodes_) if (n.is_leaf) ++c;
    return c;
}
bst_int DecisionTree::actual_depth() const {
    bst_int d = 0;
    for (const auto& n : nodes_) if (n.is_leaf) d = std::max(d, n.depth);
    return d;
}

static void dump_node(const std::vector<TreeNode>& nodes,
                      NodeId nid, std::ostringstream& oss, int indent)
{
    if (nid == kInvalidNodeId || nid >= static_cast<NodeId>(nodes.size())) return;
    const auto& n = nodes[nid];
    std::string pad(indent * 2, ' ');
    if (n.is_leaf)
        oss << pad << "leaf=" << n.leaf_value << " (n=" << n.n_samples << ")\n";
    else {
        oss << pad << "[f" << n.feature_idx << "<=" << n.split_value
            << "]  gain=" << n.split_gain << "\n";
        dump_node(nodes, n.child_left,  oss, indent + 1);
        dump_node(nodes, n.child_right, oss, indent + 1);
    }
}
std::string DecisionTree::dump_text() const {
    std::ostringstream oss;
    dump_node(nodes_, kRootNodeId, oss, 0);
    return oss.str();
}

static void dump_node_json(const std::vector<TreeNode>& nodes,
                           NodeId nid, std::ostringstream& oss, int indent)
{
    if (nid == kInvalidNodeId) return;
    const auto& n = nodes[nid];
    std::string p(indent * 2, ' ');
    oss << p << "{\n";
    oss << p << "  \"nodeid\": " << n.id << ",\n";
    oss << p << "  \"depth\": "  << n.depth << ",\n";
    if (n.is_leaf) {
        oss << p << "  \"leaf\": " << n.leaf_value << "\n";
    } else {
        oss << p << "  \"split\": \"f" << n.feature_idx << "\",\n";
        oss << p << "  \"split_condition\": " << n.split_value << ",\n";
        oss << p << "  \"gain\": " << n.split_gain << ",\n";
        oss << p << "  \"default_left\": " << (n.default_left ? "true" : "false") << ",\n";
        oss << p << "  \"children\": [\n";
        dump_node_json(nodes, n.child_left,  oss, indent + 2);
        oss << ",\n";
        dump_node_json(nodes, n.child_right, oss, indent + 2);
        oss << "\n" << p << "  ]\n";
    }
    oss << p << "}";
}
std::string DecisionTree::dump_json() const {
    std::ostringstream oss;
    dump_node_json(nodes_, kRootNodeId, oss, 0);
    return oss.str();
}

} // namespace xgb
