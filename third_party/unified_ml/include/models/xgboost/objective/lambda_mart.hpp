#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  include/objective/lambda_mart.h
//
//  Learning to Rank — LambdaMART pairwise ranking objective.
//
//  Reference: Burges et al., "Learning to Rank using Gradient Descent".
//
//  Key ideas:
//    • Samples are organised into query groups.
//    • For every pair (i, j) in a group where label[i] > label[j],
//      we compute a lambda gradient that pushes score[i] above score[j].
//    • The lambda is weighted by ΔNDCG — the change in NDCG that swapping
//      i and j would produce.
//
//  Gradient accumulation per sample i:
//    λ_i = Σ_{j: y_j < y_i} λ_{ij} - Σ_{j: y_j > y_i} λ_{ij}
//
//  where:
//    λ_{ij} = ΔZ_{ij} · σ / (1 + exp(σ(s_i - s_j)))
//
//    σ = 1 (typical)
//    ΔZ_{ij} = |ΔNDCG when rank(i) and rank(j) are swapped|
//
//  Hessian (second-order approximation):
//    h_i = Σ_j λ_{ij} * (1 - λ_{ij})
//
//  GroupStructure:
//    groups[q] = {start_row, end_row}  (half-open range)
//    Caller must sort data by query id before training.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <string>
#include <utility>

namespace xgb {

//                                       
//  GroupStructure
//  Maps query ids to contiguous row index ranges.
//                                       
struct GroupStructure {
    // Each element: {begin_index, end_index}  (half-open)
    std::vector<std::pair<bst_uint, bst_uint>> groups;

    bst_uint num_groups()  const { return static_cast<bst_uint>(groups.size()); }
    bst_uint total_rows()  const {
        return groups.empty() ? 0 : groups.back().second;
    }

    // Build from an integer group-size vector (as in XGBoost's SetGroup API).
    // group_sizes[q] = number of documents in query q.
    static GroupStructure from_sizes(const std::vector<bst_uint>& group_sizes);
};

//                                       
//  LambdaMARTObjective
//                                       
class LambdaMARTObjective {
public:
    // sigma : steepness of sigmoid (default = 1.0)
    explicit LambdaMARTObjective(bst_float sigma = 1.f);

    // Compute lambda gradients and hessians.
    //
    // scores  : current raw predictions for all rows
    // labels  : relevance grades (higher = more relevant) for all rows
    // groups  : query group structure
    // out     : output GradientPair vector (same length as scores)
    void compute_gradients(
        const std::vector<bst_float>& scores,
        const std::vector<bst_float>& labels,
        const GroupStructure&         groups,
        std::vector<GradientPair>&    out) const;

    std::string name() const { return "rank:ndcg"; }

    // Compute NDCG for one query group.
    // ranked_labels : relevance grades sorted by descending predicted score
    // k             : cutoff (0 = full list)
    static bst_float compute_ndcg(
        const std::vector<bst_float>& ranked_labels,
        bst_uint k = 0);

private:
    bst_float sigma_;   // sigmoid steepness

    //   Per-query lambda gradient computation                

    // Compute ΔNDCG weight for swapping documents at ranks ri and rj.
    static bst_float compute_delta_ndcg(
        const std::vector<bst_float>& sorted_labels,   // sorted by score desc
        bst_uint ri, bst_uint rj,
        bst_float ideal_dcg);

    // Gain of relevance grade for NDCG computation.
    static bst_float gain(bst_float label) {
        return std::pow(2.f, label) - 1.f;
    }

    // Discount at rank r (1-based).
    static bst_float discount(bst_uint rank) {
        return 1.f / std::log2(static_cast<float>(rank + 1));
    }

    // Compute ideal DCG (IDCG) for a query.
    static bst_float ideal_dcg(const std::vector<bst_float>& labels, bst_uint k);
};

} // namespace xgb
