// ═══════════════════════════════════════════════════════════════════════════
//  src/tree/leaf_wise_builder.cpp
//
//  Best-first (lossguide) tree growth implementation.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/tree/leaf_wise_builder.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace xgb {

//    Gain formulae (same as DecisionTree)                  

bst_float LeafWiseBuilder::calc_gain(double G, double H, bst_float lambda)
{
    return static_cast<bst_float>((G * G) / (H + lambda));
}

bst_float LeafWiseBuilder::calc_leaf_weight(double G, double H,
                                             bst_float lambda, bst_float alpha)
{
    // Elastic-net optimal weight: w* = -sign(G)·max(0,|G|-α)/(H+λ)
    const double abs_g   = std::abs(G);
    const double clipped = std::max(0.0, abs_g - static_cast<double>(alpha));
    const double sign_g  = (G >= 0.0) ? 1.0 : -1.0;
    return static_cast<bst_float>(-sign_g * clipped / (H + lambda));
}

bst_float LeafWiseBuilder::calc_split_gain(double GL, double HL,
                                            double GR, double HR,
                                            bst_float lambda, bst_float gamma)
{
    const bst_float g = (calc_gain(GL, HL, lambda)
                       + calc_gain(GR, HR, lambda)
                       - calc_gain(GL + GR, HL + HR, lambda)) * 0.5f - gamma;
    return g;
}

//    Constructor                               
LeafWiseBuilder::LeafWiseBuilder(std::vector<TreeNode>& tree_nodes,
                                  const TreeConfig&      cfg)
    : nodes_(tree_nodes)
    , cfg_  (cfg)
    , hist_builder_(std::make_unique<HistogramBuilder>(cfg))
    , max_leaves_(std::max(2, 1 << cfg.max_depth))   // default: 2^max_depth
{}

//    Node allocation                             
NodeId LeafWiseBuilder::alloc_node(NodeId parent, bst_int depth)
{
    const NodeId nid = static_cast<NodeId>(nodes_.size());
    TreeNode n;
    n.id     = nid;
    n.parent = parent;
    n.depth  = depth;
    n.is_leaf = true;
    nodes_.push_back(n);
    return nid;
}

void LeafWiseBuilder::make_leaf(NodeId nid, double sum_grad, double sum_hess)
{
    auto& node         = nodes_[nid];
    node.is_leaf       = true;
    node.leaf_value    = calc_leaf_weight(sum_grad, sum_hess,
                                          cfg_.lambda, cfg_.alpha);
    node.sum_grad      = sum_grad;
    node.sum_hess      = sum_hess;
}

//    Row partition                              
std::pair<std::vector<bst_uint>, std::vector<bst_uint>>
LeafWiseBuilder::partition_rows(const DMatrix&              dm,
                                 const std::vector<bst_uint>& rows,
                                 bst_uint                    feature,
                                 bst_float                   split_value,
                                 bool                        default_left) const
{
    std::vector<bst_uint> left, right;
    left.reserve(rows.size() / 2);
    right.reserve(rows.size() / 2);

    for (bst_uint row : rows) {
        // Dense DMatrix has no missing-value mask; treat as always present
        const bool missing = false;
        const bst_float val = dm.feature(row, feature);

        TreeNode dummy;
        dummy.feature_idx  = feature;
        dummy.split_value  = split_value;
        dummy.default_left = default_left;

        if (dummy.goes_left(val, missing)) {
            left.push_back(row);
        } else {
            right.push_back(row);
        }
    }
    return {std::move(left), std::move(right)};
}

//    Find best split for one leaf                      
LeafSplitCandidate
LeafWiseBuilder::find_best_split(
    NodeId                           nid,
    const DMatrix&                   dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>&     row_indices,
    const std::vector<bst_uint>&     col_indices) const
{
    LeafSplitCandidate cand;
    cand.node_id    = nid;
    cand.row_indices = row_indices;

    // Aggregate gradient stats for this node
    double sum_g = 0.0, sum_h = 0.0;
    for (bst_uint r : row_indices) {
        sum_g += grads[r].grad;
        sum_h += grads[r].hess;
    }

    if (sum_h < cfg_.min_child_weight) {
        cand.split.valid = false;
        return cand;
    }

    // Ensure cut points are built before building histograms
    hist_builder_->build_cut_points(dm, grads, 256u);

    // Build histograms using the correct API name
    auto hists = hist_builder_->build_histograms(dm, grads, row_indices, col_indices);

    // Scan each feature
    for (bst_uint fi = 0; fi < static_cast<bst_uint>(col_indices.size()); ++fi) {
        const bst_uint feat = col_indices[fi];
        if (fi >= static_cast<bst_uint>(hists.size())) continue;
        const auto& hist = hists[fi];

        double cum_g = 0.0, cum_h = 0.0;

        for (bst_uint b = 0; b + 1 < static_cast<bst_uint>(hist.bins.size()); ++b) {
            cum_g += hist.bins[b].sum_grad;
            cum_h += hist.bins[b].sum_hess;

            const double right_g = sum_g - cum_g;
            const double right_h = sum_h - cum_h;

            if (cum_h < cfg_.min_child_weight  ||
                right_h < cfg_.min_child_weight) continue;

            const bst_float gain = calc_split_gain(
                cum_g, cum_h, right_g, right_h,
                cfg_.lambda, cfg_.gamma);

            if (gain > cand.split.gain) {
                // split_value = upper_bound of this bin
                bst_float split_val = (b < static_cast<bst_uint>(hist.cut_points.size()))
                                      ? hist.cut_points[b]
                                      : static_cast<bst_float>(b);
                cand.split.gain        = gain;
                cand.split.feature_idx = feat;
                cand.split.split_value = split_val;
                cand.split.default_left = true;
                cand.split.valid        = true;
            }
        }
    }

    return cand;
}

//    Main build                                
void LeafWiseBuilder::build(
    const DMatrix&                   dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>&     row_indices,
    const std::vector<bst_uint>&     col_indices)
{
    nodes_.clear();

    // Aggregate root stats
    double root_g = 0.0, root_h = 0.0;
    for (bst_uint r : row_indices) {
        root_g += grads[r].grad;
        root_h += grads[r].hess;
    }

    // Allocate root node
    const NodeId root = alloc_node(kInvalidNodeId, 0);
    make_leaf(root, root_g, root_h);
    nodes_[root].n_samples = static_cast<bst_uint>(row_indices.size());

    //   Heap of leaf candidates                       
    using Heap = std::priority_queue<LeafSplitCandidate>;
    Heap heap;

    // Find initial split for root
    {
        auto cand = find_best_split(root, dm, grads, row_indices, col_indices);
        if (cand.split.valid && cand.split.gain > 0.f) {
            heap.push(std::move(cand));
        }
    }

    bst_int num_leaves = 1;

    //   Best-first expansion                         
    while (!heap.empty() && num_leaves < max_leaves_) {
        LeafSplitCandidate best = heap.top();
        heap.pop();

        const NodeId  nid  = best.node_id;
        auto& node = nodes_[nid];

        if (!best.split.valid || best.split.gain <= 0.f) break;
        if (node.depth >= cfg_.max_depth) continue;

        // Partition rows
        auto [left_rows, right_rows] = partition_rows(
            dm, best.row_indices,
            best.split.feature_idx,
            best.split.split_value,
            best.split.default_left);

        // Compute child stats
        double gl = 0.0, hl = 0.0;
        for (bst_uint r : left_rows)  { gl += grads[r].grad; hl += grads[r].hess; }
        double gr = root_g - gl;
        double hr = root_h - hl;

        if (hl < cfg_.min_child_weight || hr < cfg_.min_child_weight) continue;

        // Materialise split in parent node
        node.is_leaf       = false;
        node.feature_idx   = best.split.feature_idx;
        node.split_value   = best.split.split_value;
        node.default_left  = best.split.default_left;
        node.split_gain    = best.split.gain;

        // Allocate children
        const NodeId left_id  = alloc_node(nid, node.depth + 1);
        const NodeId right_id = alloc_node(nid, node.depth + 1);

        nodes_[nid].child_left  = left_id;
        nodes_[nid].child_right = right_id;

        make_leaf(left_id,  gl, hl);
        make_leaf(right_id, gr, hr);

        nodes_[left_id ].n_samples = static_cast<bst_uint>(left_rows.size());
        nodes_[right_id].n_samples = static_cast<bst_uint>(right_rows.size());

        ++num_leaves;   // one leaf replaced by two: net +1

        // Push children onto heap
        auto cand_l = find_best_split(left_id,  dm, grads, left_rows,  col_indices);
        auto cand_r = find_best_split(right_id, dm, grads, right_rows, col_indices);

        if (cand_l.split.valid && cand_l.split.gain > 0.f) heap.push(std::move(cand_l));
        if (cand_r.split.valid && cand_r.split.gain > 0.f) heap.push(std::move(cand_r));
    }
}

} // namespace xgb
