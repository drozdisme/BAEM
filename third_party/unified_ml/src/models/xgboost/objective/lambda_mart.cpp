// ═══════════════════════════════════════════════════════════════════════════
//  src/objective/lambda_mart.cpp
//
//  LambdaMART pairwise ranking gradients with ΔNDCG weighting.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/objective/lambda_mart.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace xgb {

//    GroupStructure factory                         
GroupStructure GroupStructure::from_sizes(
    const std::vector<bst_uint>& group_sizes)
{
    GroupStructure gs;
    gs.groups.reserve(group_sizes.size());
    bst_uint offset = 0;
    for (bst_uint sz : group_sizes) {
        gs.groups.push_back({offset, offset + sz});
        offset += sz;
    }
    return gs;
}

//    Constructor                               
LambdaMARTObjective::LambdaMARTObjective(bst_float sigma)
    : sigma_(sigma) {}

//    Ideal DCG                                
bst_float LambdaMARTObjective::ideal_dcg(
    const std::vector<bst_float>& labels, bst_uint k)
{
    // Sort labels descending to compute ideal ranking
    std::vector<bst_float> sorted = labels;
    std::sort(sorted.begin(), sorted.end(), std::greater<bst_float>());

    const bst_uint limit = (k == 0 || k > sorted.size())
                           ? static_cast<bst_uint>(sorted.size()) : k;

    bst_float dcg = 0.f;
    for (bst_uint r = 0; r < limit; ++r) {
        dcg += gain(sorted[r]) * discount(r + 1);
    }
    return dcg;
}

//    NDCG for one query                           
bst_float LambdaMARTObjective::compute_ndcg(
    const std::vector<bst_float>& ranked_labels, bst_uint k)
{
    if (ranked_labels.empty()) return 0.f;

    const bst_uint limit = (k == 0 || k > ranked_labels.size())
                           ? static_cast<bst_uint>(ranked_labels.size()) : k;

    bst_float dcg = 0.f;
    for (bst_uint r = 0; r < limit; ++r) {
        dcg += gain(ranked_labels[r]) * discount(r + 1);
    }

    const bst_float idcg = ideal_dcg(ranked_labels, k);
    if (idcg == 0.f) return 0.f;
    return dcg / idcg;
}

//    ΔNDCG for swapping documents at ranks ri and rj            
//  ranked_labels: labels of documents sorted by current predicted score (desc).
//  ri, rj: 0-based positions in that order (ri < rj).
//  Returns absolute value of the NDCG change.
bst_float LambdaMARTObjective::compute_delta_ndcg(
    const std::vector<bst_float>& sorted_labels,
    bst_uint ri, bst_uint rj,
    bst_float idcg)
{
    if (idcg <= 0.f) return 0.f;

    const bst_float g_i = gain(sorted_labels[ri]);
    const bst_float g_j = gain(sorted_labels[rj]);
    const bst_float d_i = discount(ri + 1);   // 1-based rank
    const bst_float d_j = discount(rj + 1);

    // ΔDCG = (g_i - g_j)(d_j - d_i)
    //  Since sorted by score desc and y_i > y_j: g_i > g_j, d_i > d_j
    //  Swapping makes DCG worse → |ΔDCG| = (g_i - g_j)(d_i - d_j)
    const bst_float delta = std::abs((g_i - g_j) * (d_i - d_j));
    return delta / idcg;
}

//    Main gradient computation                        
void LambdaMARTObjective::compute_gradients(
    const std::vector<bst_float>& scores,
    const std::vector<bst_float>& labels,
    const GroupStructure&         groups,
    std::vector<GradientPair>&    out) const
{
    const bst_uint n = static_cast<bst_uint>(scores.size());
    assert(labels.size() == n);

    out.assign(n, GradientPair{0.f, 0.f});

    for (const auto& [begin, end] : groups.groups) {
        const bst_uint group_size = end - begin;
        if (group_size < 2) continue;

        //   Sort document indices by score descending            
        std::vector<bst_uint> order(group_size);
        std::iota(order.begin(), order.end(), begin);
        std::sort(order.begin(), order.end(), [&](bst_uint a, bst_uint b) {
            return scores[a] > scores[b];
        });

        //   Extract relevance grades in score-sorted order         
        std::vector<bst_float> sorted_labels(group_size);
        for (bst_uint k = 0; k < group_size; ++k) {
            sorted_labels[k] = labels[order[k]];
        }

        //   Ideal DCG for this query                     
        const bst_float idcg = ideal_dcg(sorted_labels, 0);

        //   Iterate over all pairs (i, j) where label[i] > label[j]     
        for (bst_uint ri = 0; ri < group_size; ++ri) {
            for (bst_uint rj = ri + 1; rj < group_size; ++rj) {
                const bst_uint gi = order[ri];   // global row index
                const bst_uint gj = order[rj];

                const bst_float yi = labels[gi];
                const bst_float yj = labels[gj];

                if (yi == yj) continue;   // same relevance — no pair

                // Ensure yi > yj  (i should rank above j)
                const bst_uint hi = (yi > yj) ? gi : gj;
                const bst_uint lo = (yi > yj) ? gj : gi;
                const bst_uint r_hi = (yi > yj) ? ri : rj;
                const bst_uint r_lo = (yi > yj) ? rj : ri;

                // ΔNDCG for this pair
                const bst_float delta = compute_delta_ndcg(
                    sorted_labels,
                    std::min(r_hi, r_lo),
                    std::max(r_hi, r_lo),
                    idcg);

                // Sigmoid of score difference: σ(s_hi - s_lo)
                const bst_float score_diff = scores[hi] - scores[lo];
                const bst_float sigmoid = 1.f / (1.f + std::exp(-sigma_ * score_diff));

                // λ_{hi,lo} = delta * (1 - sigmoid)   [for the higher doc]
                // gradient: push hi up, lo down
                const bst_float lambda_ij = delta * (1.f - sigmoid);

                out[hi].grad -= lambda_ij;    // want hi to go up → negative grad
                out[lo].grad += lambda_ij;    // want lo to go down → positive grad

                // Hessian approximation: |λ_ij * sigmoid * (1 - sigmoid)| * σ
                const bst_float h = std::max(
                    sigma_ * lambda_ij * sigmoid * (1.f - sigmoid),
                    1e-6f);
                out[hi].hess += h;
                out[lo].hess += h;
            }
        }
    }
}

} // namespace xgb
