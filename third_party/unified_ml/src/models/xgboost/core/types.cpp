#include "models/xgboost/core/types.hpp"
#include <numeric>

namespace xgb {

TreeNodeStats::TreeNodeStats(const std::vector<GradientPair>& grads,
                             const std::vector<bst_uint>& indices) {
    for (bst_uint idx : indices) {
        sum_grad += grads[idx].grad;
        sum_hess += grads[idx].hess;
        ++n_samples;
    }
}

} // namespace xgb
