#include "models/xgboost/objective/logistic_loss.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace xgb {

void LogisticLoss::compute_gradients(
    const std::vector<bst_float>& scores,
    const std::vector<bst_float>& labels,
    std::vector<GradientPair>& out) const
{
    out.resize(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        bst_float p    = sigmoid(scores[i]);
        out[i].grad    = p - labels[i];
        out[i].hess    = std::max(p * (1.f - p), min_hess_);
    }
}

bst_float LogisticLoss::compute_loss(
    const std::vector<bst_float>& scores,
    const std::vector<bst_float>& labels) const
{
    // Binary cross-entropy
    double loss = 0.0;
    constexpr bst_float eps = 1e-7f;
    for (size_t i = 0; i < scores.size(); ++i) {
        bst_float p = sigmoid(scores[i]);
        p = std::max(eps, std::min(1.f - eps, p));
        loss -= labels[i] * std::log(p)
                + (1.f - labels[i]) * std::log(1.f - p);
    }
    return static_cast<bst_float>(loss / scores.size());
}

bst_float LogisticLoss::base_score(
    const std::vector<bst_float>& labels) const
{
    // Mean positive rate → clip and logit
    if (labels.empty()) return 0.f;
    double mean = 0.0;
    for (bst_float v : labels) mean += v;
    mean /= labels.size();
    mean = std::max(1e-5, std::min(1.0 - 1e-5, mean));
    // logit(p) = log(p / (1-p))
    return static_cast<bst_float>(std::log(mean / (1.0 - mean)));
}

} // namespace xgb
