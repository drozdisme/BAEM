#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <limits>

namespace xgb {

//                        
//  Scalar aliases
//                        
using bst_float  = float;       // feature / prediction values
using bst_uint   = uint32_t;    // row / column indices
using bst_ulong  = uint64_t;
using bst_int    = int32_t;

//                        
//  GradientPair  (gi, hi)
//  First-order gradient + second-order Hessian
//  for a single instance.
//  Sec. 2.2 of the paper: L̃(t) = Σ[gi·ft(xi) + ½·hi·ft²(xi)]
//                        
struct GradientPair {
    bst_float grad{0.f};   // gi  — first-order gradient
    bst_float hess{0.f};   // hi  — second-order Hessian

    GradientPair() = default;
    GradientPair(bst_float g, bst_float h) : grad(g), hess(h) {}

    GradientPair& operator+=(const GradientPair& rhs) {
        grad += rhs.grad;
        hess += rhs.hess;
        return *this;
    }
    GradientPair operator+(const GradientPair& rhs) const {
        return {grad + rhs.grad, hess + rhs.hess};
    }
    GradientPair operator-(const GradientPair& rhs) const {
        return {grad - rhs.grad, hess - rhs.hess};
    }
};

//                        
//  TreeNodeStats
//  Aggregate gradient statistics for a node.
//  Used in split scoring (Eq. 6 of the paper).
//                        
struct TreeNodeStats {
    double sum_grad{0.0};
    double sum_hess{0.0};
    bst_uint n_samples{0};

    TreeNodeStats() = default;
    explicit TreeNodeStats(const std::vector<GradientPair>& grads,
                           const std::vector<bst_uint>& indices);

    void add(const GradientPair& gp) {
        sum_grad += gp.grad;
        sum_hess += gp.hess;
        ++n_samples;
    }

    TreeNodeStats operator-(const TreeNodeStats& rhs) const {
        TreeNodeStats s;
        s.sum_grad  = sum_grad  - rhs.sum_grad;
        s.sum_hess  = sum_hess  - rhs.sum_hess;
        s.n_samples = n_samples - rhs.n_samples;
        return s;
    }
};

//                        
//  SplitCandidate
//  Stores the best split found for a node.
//  Corresponds to Alg. 1 / Eq. 7 in the paper.
//                        
struct SplitCandidate {
    bst_float  gain         {-std::numeric_limits<bst_float>::infinity()};
    bst_uint   feature_idx  {0};
    bst_float  split_value  {0.f};
    bool       default_left {true};   // direction when value is missing (Sec. 3.4)
    bool       valid        {false};

    bool operator<(const SplitCandidate& rhs) const {
        if (gain != rhs.gain) return gain < rhs.gain;
        return feature_idx > rhs.feature_idx;  // lower index wins on tie
    }
    bool operator>(const SplitCandidate& rhs) const {
        if (gain != rhs.gain) return gain > rhs.gain;
        return feature_idx < rhs.feature_idx;  // lower index wins on tie
    }
};

//                        
//  RegTree typedefs (forward declarations)
//                        
using NodeId = bst_int;
constexpr NodeId kInvalidNodeId = -1;
constexpr NodeId kRootNodeId    =  0;

//                        
//  Task type
//                        
enum class TaskType {
    kRegression,
    kBinaryClassification,
    kMultiClassification,
    kRanking
};

//                        
//  Prediction output
//                        
struct PredictionResult {
    std::vector<bst_float> scores;   // raw margin scores
    std::vector<bst_float> probs;    // post-transform probabilities
};

} // namespace xgb
