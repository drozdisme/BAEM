#pragma once
//                                        
//  WeightedQuantileSketch   (XGBoost paper §3.3 / Greenwald-Khanna adapted)
//
//  Builds a compact summary of (value, weight) pairs that answers approximate
//  quantile queries with error ≤ ε·W (W = total Hessian mass).
//
//  Summary invariant (GK):
//    For any rank query φ ∈ [0, W] there exists entry e s.t.
//      rmin(e) ≤ φ ≤ rmax(e)  AND  rmax(e) - rmin(e) ≤ 2·ε·W
//
//  API:
//    sketch.push(value, weight)   — stream samples one by one
//    sketch.finalize()            — consolidate buffer into summary, prune
//    sketch.merge(other)          — combine two finalized sketches
//    sketch.get_cut_points(n)     — extract ≤ n quantile boundaries
//                                        
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <algorithm>
#include <iterator>   // std::back_inserter
#include <cmath>
#include <limits>
#include <utility>    // std::pair

namespace xgb {

// Explicit comparator struct — avoids MSVC lambda deduction issues with std::merge
struct PairLessFirst {
    bool operator()(const std::pair<bst_float,bst_float>& a,
                    const std::pair<bst_float,bst_float>& b) const {
        return a.first < b.first;
    }
};

class WeightedQuantileSketch {
public:
    //    Summary entry                           
    struct Entry {
        bst_float value;   // representative feature value
        bst_float rmin;    // minimum cumulative weight before this entry
        bst_float rmax;    // maximum cumulative weight up to this entry
        bst_float wmin;    // weight contributed by this entry

        bool operator<(const Entry& o) const { return value < o.value; }
    };

    explicit WeightedQuantileSketch(bst_float eps = 0.05f)
        : eps_(eps), total_weight_(0.f) {}

    //    Streaming                             
    void push(bst_float value, bst_float weight) {
        if (weight <= 0.f) return;
        buffer_.emplace_back(value, weight);
        total_weight_ += weight;
    }

    // Consolidate buffer into summary and prune.
    // Must be called before merge() or get_cut_points().
    void finalize() {
        if (buffer_.empty()) return;

        // Sort ascending by value
        std::stable_sort(buffer_.begin(), buffer_.end(), PairLessFirst());

        // Merge identical values
        std::vector<std::pair<bst_float,bst_float>> compact;
        compact.reserve(buffer_.size());
        for (std::size_t i = 0; i < buffer_.size(); ++i) {
            bst_float v = buffer_[i].first;
            bst_float w = buffer_[i].second;
            if (!compact.empty() && compact.back().first == v)
                compact.back().second += w;
            else
                compact.emplace_back(v, w);
        }
        buffer_.clear();

        // Extract existing summary as (value, wmin) pairs
        std::vector<std::pair<bst_float,bst_float>> existing;
        existing.reserve(summary_.size());
        for (std::size_t i = 0; i < summary_.size(); ++i)
            existing.emplace_back(summary_[i].value, summary_[i].wmin);
        summary_.clear();

        // Merge-sort the two sorted sequences
        std::vector<std::pair<bst_float,bst_float>> merged;
        merged.reserve(existing.size() + compact.size());
        std::merge(existing.begin(), existing.end(),
                   compact.begin(),  compact.end(),
                   std::back_inserter(merged),
                   PairLessFirst());

        // Deduplicate equal values
        std::vector<std::pair<bst_float,bst_float>> deduped;
        deduped.reserve(merged.size());
        for (std::size_t i = 0; i < merged.size(); ++i) {
            bst_float v = merged[i].first;
            bst_float w = merged[i].second;
            if (!deduped.empty() && deduped.back().first == v)
                deduped.back().second += w;
            else
                deduped.emplace_back(v, w);
        }

        build_from_pairs(deduped);
        prune();
    }

    //    Merge two finalized sketches                    
    void merge(const WeightedQuantileSketch& other) {
        if (other.summary_.empty() && other.buffer_.empty()) return;
        total_weight_ += other.total_weight_;

        std::vector<std::pair<bst_float,bst_float>> pA, pB;
        pA.reserve(summary_.size());
        pB.reserve(other.summary_.size());
        for (std::size_t i = 0; i < summary_.size(); ++i)
            pA.emplace_back(summary_[i].value, summary_[i].wmin);
        for (std::size_t i = 0; i < other.summary_.size(); ++i)
            pB.emplace_back(other.summary_[i].value, other.summary_[i].wmin);
        summary_.clear();

        std::vector<std::pair<bst_float,bst_float>> merged;
        merged.reserve(pA.size() + pB.size());
        std::merge(pA.begin(), pA.end(),
                   pB.begin(), pB.end(),
                   std::back_inserter(merged),
                   PairLessFirst());

        // Deduplicate
        std::vector<std::pair<bst_float,bst_float>> deduped;
        deduped.reserve(merged.size());
        for (std::size_t i = 0; i < merged.size(); ++i) {
            bst_float v = merged[i].first;
            bst_float w = merged[i].second;
            if (!deduped.empty() && deduped.back().first == v)
                deduped.back().second += w;
            else
                deduped.emplace_back(v, w);
        }
        build_from_pairs(deduped);
        prune();
    }

    //    Extract cut points                         
    std::vector<bst_float> get_cut_points(bst_uint n_bins) const {
        if (summary_.empty() || total_weight_ <= 0.f) return {};

        std::vector<bst_float> cuts;
        cuts.reserve(n_bins + 1);

        const bst_float W        = total_weight_;
        const bst_float bin_size = W / static_cast<bst_float>(n_bins);

        bst_float next_boundary = bin_size;
        bst_float last_emitted  = -std::numeric_limits<bst_float>::max();

        for (std::size_t i = 0; i < summary_.size(); ++i) {
            const Entry& e = summary_[i];
            while (e.rmax >= next_boundary && next_boundary < W) {
                if (e.value != last_emitted) {
                    cuts.push_back(e.value);
                    last_emitted = e.value;
                }
                next_boundary += bin_size;
            }
        }

        // Always include max value
        const bst_float max_val = summary_.back().value;
        if (cuts.empty() || cuts.back() != max_val)
            cuts.push_back(max_val);

        // Deduplicate
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        return cuts;
    }

    //    Accessors                             
    bst_float                  total_weight() const { return total_weight_; }
    const std::vector<Entry>&  summary()      const { return summary_; }
    bst_uint summary_size() const { return static_cast<bst_uint>(summary_.size()); }

    void clear() { summary_.clear(); buffer_.clear(); total_weight_ = 0.f; }

private:
    bst_float eps_;
    bst_float total_weight_;
    std::vector<std::pair<bst_float,bst_float>> buffer_;
    std::vector<Entry> summary_;

    void build_from_pairs(const std::vector<std::pair<bst_float,bst_float>>& pairs) {
        bst_float cumulative = 0.f;
        summary_.reserve(pairs.size());
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            Entry e;
            e.value = pairs[i].first;
            e.wmin  = pairs[i].second;
            e.rmin  = cumulative;
            e.rmax  = cumulative + e.wmin;
            summary_.push_back(e);
            cumulative += e.wmin;
        }
    }

    void prune() {
        const bst_uint  tgt         = target_size();
        const bst_float error_bound = 2.f * eps_ * total_weight_;

        while (summary_.size() > tgt && summary_.size() > 2) {
            bst_uint  best_i   = 0;
            bst_float best_err = std::numeric_limits<bst_float>::max();
            bool      found    = false;

            for (bst_uint i = 1; i + 1 < static_cast<bst_uint>(summary_.size()); ++i) {
                bst_float new_err = summary_[i + 1].rmax - summary_[i].rmin;
                if (new_err <= error_bound && new_err < best_err) {
                    best_err = new_err;
                    best_i   = i;
                    found    = true;
                }
            }

            if (!found) break;

            // Merge entry best_i into best_i+1
            summary_[best_i + 1].rmin = summary_[best_i].rmin;
            summary_.erase(summary_.begin() + best_i);
        }
    }

    bst_uint target_size() const {
        return static_cast<bst_uint>(std::ceil(1.0f / (2.0f * eps_))) + 2;
    }
};

} // namespace xgb
