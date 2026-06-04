// ═══════════════════════════════════════════════════════════════════════════
//  src/objective/multi_softmax.cpp
//
//  Multiclass softmax gradient computation.
// ═══════════════════════════════════════════════════════════════════════════
#include "models/xgboost/objective/multi_softmax.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xgb {

MultiSoftmaxObjective::MultiSoftmaxObjective(bst_int num_class)
    : num_class_(num_class)
{
    if (num_class_ < 2) {
        throw std::invalid_argument(
            "MultiSoftmaxObjective: num_class must be >= 2");
    }
}

//    Numerically stable softmax over one sample window           
void MultiSoftmaxObjective::softmax_window(const bst_float* scores_begin,
                                            bst_float*       probs_begin) const
{
    // Step 1: find max for numerical stability  (log-sum-exp trick)
    bst_float max_s = scores_begin[0];
    for (bst_int c = 1; c < num_class_; ++c) {
        if (scores_begin[c] > max_s) max_s = scores_begin[c];
    }

    // Step 2: exp(s - max)
    bst_float sum = 0.f;
    for (bst_int c = 0; c < num_class_; ++c) {
        probs_begin[c] = std::exp(scores_begin[c] - max_s);
        sum += probs_begin[c];
    }

    // Step 3: normalise
    const bst_float inv_sum = 1.f / sum;
    for (bst_int c = 0; c < num_class_; ++c) {
        probs_begin[c] *= inv_sum;
    }
}

//    Compute gradients                           
//
//  scores layout: [n*C]  i.e. sample0_class0, sample0_class1, ..., sample1_class0 ...
//  out layout:    same
//
void MultiSoftmaxObjective::compute_gradients(
    const std::vector<bst_float>& scores,
    const std::vector<bst_float>& labels,
    std::vector<GradientPair>&    out) const
{
    const bst_uint n = static_cast<bst_uint>(labels.size());
    assert(scores.size() == static_cast<size_t>(n) * num_class_);

    out.resize(static_cast<size_t>(n) * num_class_);

    // Temporary probability buffer (one sample at a time)
    std::vector<bst_float> probs(num_class_);

    for (bst_uint i = 0; i < n; ++i) {
        const bst_float* s = scores.data() + i * num_class_;
        bst_float*       p = probs.data();

        softmax_window(s, p);

        const bst_int y = static_cast<bst_int>(labels[i]);
        if (y < 0 || y >= num_class_) {
            throw std::runtime_error(
                "MultiSoftmaxObjective: label out of range");
        }

        GradientPair* gp = out.data() + i * num_class_;
        for (bst_int c = 0; c < num_class_; ++c) {
            const bst_float pc = p[c];
            // g_{ic} = p_{ic} - 1[y_i == c]
            // h_{ic} = p_{ic} * (1 - p_{ic})   clipped to [1e-6, 1]
            gp[c].grad = pc - (c == y ? 1.f : 0.f);
            gp[c].hess = std::max(pc * (1.f - pc), 1e-6f);
        }
    }
}

//    Batch softmax transform                        
std::vector<bst_float>
MultiSoftmaxObjective::softmax_transform(
    const std::vector<bst_float>& scores) const
{
    const bst_uint n = static_cast<bst_uint>(scores.size()) / num_class_;
    std::vector<bst_float> probs(scores.size());

    for (bst_uint i = 0; i < n; ++i) {
        softmax_window(scores.data()  + i * num_class_,
                       probs.data()   + i * num_class_);
    }
    return probs;
}

//    Argmax prediction                           
std::vector<bst_int>
MultiSoftmaxObjective::predict_class(
    const std::vector<bst_float>& scores) const
{
    const bst_uint n = static_cast<bst_uint>(scores.size()) / num_class_;
    std::vector<bst_int> preds(n);

    for (bst_uint i = 0; i < n; ++i) {
        const bst_float* s = scores.data() + i * num_class_;
        preds[i] = static_cast<bst_int>(
            std::max_element(s, s + num_class_) - s);
    }
    return preds;
}

//    Cross-entropy loss (log of softmax)                  
bst_float MultiSoftmaxObjective::compute_loss(
    const std::vector<bst_float>& scores,
    const std::vector<bst_float>& labels) const
{
    const bst_uint n = static_cast<bst_uint>(labels.size());
    std::vector<bst_float> probs(num_class_);
    bst_float loss = 0.f;

    for (bst_uint i = 0; i < n; ++i) {
        const bst_float* s = scores.data() + i * num_class_;
        softmax_window(s, probs.data());
        const bst_int y = static_cast<bst_int>(labels[i]);
        loss -= std::log(std::max(probs[y], 1e-12f));
    }
    return loss / static_cast<bst_float>(n);
}

} // namespace xgb
