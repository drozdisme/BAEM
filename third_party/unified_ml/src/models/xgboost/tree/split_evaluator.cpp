#include "models/xgboost/tree/split_evaluator.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace xgb {

//              
//  L1/L2 scoring helpers  (Eq. 6 + elastic-net extension)
//              

bst_float SplitEvaluator::node_score(double G, double H) const {
  double denom = H + cfg_.lambda;
  if (denom < 1e-9) return 0.f;
  if (cfg_.alpha > 0.f) {
    double absG    = std::fabs(G);
    double soft_thresh = absG - static_cast<double>(cfg_.alpha);
    if (soft_thresh <= 0.0) return 0.f;
    return static_cast<bst_float>(soft_thresh * soft_thresh / denom);
  }
  return static_cast<bst_float>((G * G) / denom);
}

bst_float SplitEvaluator::split_gain(
  double GL, double HL, double GR, double HR) const
{
  return 0.5f * (node_score(GL, HL) + node_score(GR, HR)
       - node_score(GL + GR, HL + HR))
     - cfg_.gamma;
}

bst_float SplitEvaluator::score_split(
  double GL, double HL, double GR, double HR) const
{
  return split_gain(GL, HL, GR, HR);
}

//              
//  Feature 1: Parallelised Exact Greedy Split (Algorithm 1 + OpenMP)
//
//  The outer loop over features is embarrassingly parallel: each feature's
//  best split can be computed independently. We exploit this with:
//
//  #pragma omp parallel for reduction(max:best)
//
//  where the reduction uses operator> defined on SplitCandidate.
//
//  Thread count is controlled by BoosterConfig::nthread (-1 = all cores).
//
//  Implementation notes:
//  - Each thread maintains a thread-local `local_best`.
//  - At the barrier, thread-locals are merged via the reduction.
//  - The inner per-feature sort is left sequential (it's O(n log n) and
//  typically n << n_features, so the overhead of nested parallelism
//  would exceed the benefit).
//  - No mutex is needed: each thread writes only to its own local_best.
//
//  Reference: §4.1 of Chen & Guestrin — "column blocks for parallel learning".
//              
SplitCandidate SplitEvaluator::find_best_split(
  const DMatrix& dm,
  const std::vector<GradientPair>& grads,
  const std::vector<bst_uint>& row_indices,
  const std::vector<bst_uint>& col_indices) const
{
  if (cfg_.use_approx) {
    return find_best_split_approx(dm, grads, row_indices, col_indices);
  }

  // Aggregate total G and H for this node (serial — O(n), fast)
  double G_total = 0.0, H_total = 0.0;
  for (bst_uint i : row_indices) {
    G_total += grads[i].grad;
    H_total += grads[i].hess;
  }

  SplitCandidate best;

  // Feature 1: Set thread count from TreeConfig::nthread    
  // nthread == -1 → use all available cores (OpenMP default behaviour).
  // Any positive value caps parallelism, useful for memory-constrained envs.
#ifdef _OPENMP
  if (cfg_.nthread > 0) {
    omp_set_num_threads(cfg_.nthread);
  }
  // nthread == -1: leave OpenMP default (OMP_NUM_THREADS env var or all cores)
#endif

  // col_indices to a contiguous array for OpenMP indexing   
  const int n_cols = static_cast<int>(col_indices.size());

  // Thread-local bests collected after the parallel region
  // We use a manual reduction rather than a custom OpenMP declare reduction
  // to maintain C++14 compatibility (declare reduction requires C++17 on
  // some compilers with complex initialiser).
  std::vector<SplitCandidate> thread_bests;

#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    {
    thread_bests.resize(omp_get_num_threads());
    }

#pragma omp for schedule(static)
    for (int ci = 0; ci < n_cols; ++ci) {
    bst_uint feat = col_indices[ci];
    SplitCandidate local_best;

    if (cfg_.sparsity_aware) {
      local_best = find_best_split_sparse(
        dm, grads, row_indices, feat, G_total, H_total);
    } else {
      // Sort a thread-local copy of row indices by feature value
      std::vector<bst_uint> sorted_rows = row_indices;
      std::sort(sorted_rows.begin(), sorted_rows.end(),
        [&](bst_uint a, bst_uint b) {
        return dm.feature(a, feat) < dm.feature(b, feat);
        });

      double GL = 0.0, HL = 0.0;
      for (size_t j = 0; j + 1 < sorted_rows.size(); ++j) {
        bst_uint idx = sorted_rows[j];
        GL += grads[idx].grad;
        HL += grads[idx].hess;

        bst_float cur_val  = dm.feature(sorted_rows[j],   feat);
        bst_float next_val = dm.feature(sorted_rows[j + 1], feat);
        if (cur_val == next_val) continue;

        double GR = G_total - GL;
        double HR = H_total - HL;

        if (HL < cfg_.min_child_weight) continue;
        if (HR < cfg_.min_child_weight) continue;

        bst_float gain = split_gain(GL, HL, GR, HR);
        if (gain > local_best.gain) {
        local_best.gain   = gain;
        local_best.feature_idx  = feat;
        local_best.split_value  = (cur_val + next_val) * 0.5f;
        local_best.default_left = true;
        local_best.valid    = true;
        }
      }
    }

    // Write thread-local best into the per-thread slot
    int tid = omp_get_thread_num();
    if (local_best.gain > thread_bests[tid].gain)
      thread_bests[tid] = local_best;
    }
  }

  // Serial reduction over thread-local bests
  for (const auto& tb : thread_bests) {
        if (tb.gain > best.gain) best = tb;
    }

#else
    //   Serial fallback when OpenMP is not available            
    for (int ci = 0; ci < n_cols; ++ci) {
        bst_uint feat = col_indices[ci];

        if (cfg_.sparsity_aware) {
            SplitCandidate c = find_best_split_sparse(
                dm, grads, row_indices, feat, G_total, H_total);
            if (c.gain > best.gain) best = c;
        } else {
            std::vector<bst_uint> sorted_rows = row_indices;
            std::sort(sorted_rows.begin(), sorted_rows.end(),
                [&](bst_uint a, bst_uint b_) {
                    return dm.feature(a, feat) < dm.feature(b_, feat);
                });

            double GL = 0.0, HL = 0.0;
            for (size_t j = 0; j + 1 < sorted_rows.size(); ++j) {
                bst_uint idx = sorted_rows[j];
                GL += grads[idx].grad;
                HL += grads[idx].hess;

                bst_float cur_val  = dm.feature(sorted_rows[j],     feat);
                bst_float next_val = dm.feature(sorted_rows[j + 1], feat);
                if (cur_val == next_val) continue;

                double GR = G_total - GL;
                double HR = H_total - HL;

                if (HL < cfg_.min_child_weight) continue;
                if (HR < cfg_.min_child_weight) continue;

                bst_float gain = split_gain(GL, HL, GR, HR);
                if (gain > best.gain) {
                    best.gain         = gain;
                    best.feature_idx  = feat;
                    best.split_value  = (cur_val + next_val) * 0.5f;
                    best.default_left = true;
                    best.valid        = true;
                }
            }
        }
    }
#endif

    return best;
}

//                                        
//  Approximate split (Algorithm 2)  — parallelised over features as well
//                                        
SplitCandidate SplitEvaluator::find_best_split_approx(
    const DMatrix& dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>& row_indices,
    const std::vector<bst_uint>& col_indices) const
{
    HistogramBuilder builder(cfg_);
    builder.build_cut_points(dm, grads, 256u);

    std::vector<FeatureHistogram> hists =
        builder.build_histograms(dm, grads, row_indices, col_indices);

    double G_total = 0.0, H_total = 0.0;
    for (bst_uint i : row_indices) {
        G_total += grads[i].grad;
        H_total += grads[i].hess;
    }

    SplitCandidate best;
    const int n_cols = static_cast<int>(col_indices.size());
    std::vector<SplitCandidate> thread_bests;

#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        { thread_bests.resize(omp_get_num_threads()); }

#pragma omp for schedule(static)
        for (int ci = 0; ci < n_cols; ++ci) {
            bst_uint feat = col_indices[ci];
            const FeatureHistogram& hist = hists[feat];
            const bst_uint n_bins = hist.n_bins();
            SplitCandidate local_best;

            double GL = 0.0, HL = 0.0;
            for (bst_uint b = 0; b + 1 < n_bins; ++b) {
                GL += hist.bins[b].sum_grad;
                HL += hist.bins[b].sum_hess;
                double GR = G_total - GL;
                double HR = H_total - HL;

                if (HL < cfg_.min_child_weight) continue;
                if (HR < cfg_.min_child_weight) continue;

                bst_float split_val = (b < static_cast<bst_uint>(hist.cut_points.size()))
                                      ? hist.cut_points[b]
                                      : std::numeric_limits<bst_float>::max();

                bst_float gain = split_gain(GL, HL, GR, HR);
                if (gain > local_best.gain) {
                    local_best.gain         = gain;
                    local_best.feature_idx  = feat;
                    local_best.split_value  = split_val;
                    local_best.default_left = (G_total - GL > 0.0);
                    local_best.valid        = true;
                }
            }

            int tid = omp_get_thread_num();
            if (local_best.gain > thread_bests[tid].gain)
                thread_bests[tid] = local_best;
        }
    }
    for (const auto& tb : thread_bests)
        if (tb.gain > best.gain) best = tb;

#else
    for (int ci = 0; ci < n_cols; ++ci) {
        bst_uint feat = col_indices[ci];
        const FeatureHistogram& hist = hists[feat];
        const bst_uint n_bins = hist.n_bins();

        double GL = 0.0, HL = 0.0;
        for (bst_uint b = 0; b + 1 < n_bins; ++b) {
            GL += hist.bins[b].sum_grad;
            HL += hist.bins[b].sum_hess;
            double GR = G_total - GL;
            double HR = H_total - HL;

            if (HL < cfg_.min_child_weight) continue;
            if (HR < cfg_.min_child_weight) continue;

            bst_float split_val = (b < static_cast<bst_uint>(hist.cut_points.size()))
                                  ? hist.cut_points[b]
                                  : std::numeric_limits<bst_float>::max();

            bst_float gain = split_gain(GL, HL, GR, HR);
            if (gain > best.gain) {
                best.gain         = gain;
                best.feature_idx  = feat;
                best.split_value  = split_val;
                best.default_left = (G_total - GL > 0.0);
                best.valid        = true;
            }
        }
    }
#endif

    return best;
}

//   Sparsity-aware split (Algorithm 3)                     
SplitCandidate SplitEvaluator::find_best_split_sparse(
    const DMatrix& dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>& row_indices,
    bst_uint feature_idx,
    double G_total, double H_total) const
{
    std::vector<bst_uint> present = row_indices;
    std::sort(present.begin(), present.end(),
        [&](bst_uint a, bst_uint b) {
            return dm.feature(a, feature_idx) < dm.feature(b, feature_idx);
        });

    SplitCandidate best;

    // Pass 1: missing → right (ascending scan)
    {
        double GL = 0.0, HL = 0.0;
        for (size_t j = 0; j + 1 < present.size(); ++j) {
            GL += grads[present[j]].grad;
            HL += grads[present[j]].hess;

            bst_float cur_val  = dm.feature(present[j],     feature_idx);
            bst_float next_val = dm.feature(present[j + 1], feature_idx);
            if (cur_val == next_val) continue;

            double GR = G_total - GL, HR = H_total - HL;
            if (HL < cfg_.min_child_weight) continue;
            if (HR < cfg_.min_child_weight) continue;

            bst_float gain = split_gain(GL, HL, GR, HR);
            if (gain > best.gain) {
                best = { gain, feature_idx,
                         (cur_val + next_val) * 0.5f,
                         /*default_left=*/false, /*valid=*/true };
            }
        }
    }

    // Pass 2: missing → left (descending scan)
    {
        double GR = 0.0, HR = 0.0;
        for (int j = static_cast<int>(present.size()) - 1; j >= 1; --j) {
            GR += grads[present[j]].grad;
            HR += grads[present[j]].hess;

            bst_float cur_val  = dm.feature(present[j],     feature_idx);
            bst_float prev_val = dm.feature(present[j - 1], feature_idx);
            if (cur_val == prev_val) continue;

            double GL = G_total - GR, HL = H_total - HR;
            if (HL < cfg_.min_child_weight) continue;
            if (HR < cfg_.min_child_weight) continue;

            bst_float gain = split_gain(GL, HL, GR, HR);
            if (gain > best.gain) {
                best = { gain, feature_idx,
                         (prev_val + cur_val) * 0.5f,
                         /*default_left=*/true, /*valid=*/true };
            }
        }
    }

    return best;
}

//   find_best_split_approx_from_hists                     
SplitCandidate SplitEvaluator::find_best_split_approx_from_hists(
    const std::vector<FeatureHistogram>& hists,
    double G_total,
    double H_total,
    const std::vector<bst_uint>& col_indices) const
{
    SplitCandidate best;
    const int n_cols = static_cast<int>(col_indices.size());
    std::vector<SplitCandidate> thread_bests;

#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        { thread_bests.resize(omp_get_num_threads()); }

#pragma omp for schedule(static)
        for (int ci = 0; ci < n_cols; ++ci) {
            bst_uint feat = col_indices[ci];
            if (feat >= static_cast<bst_uint>(hists.size())) continue;
            const FeatureHistogram& hist = hists[feat];
            const bst_uint n_bins = hist.n_bins();
            SplitCandidate local_best;

            double GL = 0.0, HL = 0.0;
            for (bst_uint b = 0; b + 1 < n_bins; ++b) {
                GL += hist.bins[b].sum_grad;
                HL += hist.bins[b].sum_hess;
                double GR = G_total - GL, HR = H_total - HL;

                if (HL < cfg_.min_child_weight) continue;
                if (HR < cfg_.min_child_weight) continue;

                bst_float split_val = (b < static_cast<bst_uint>(hist.cut_points.size()))
                                      ? hist.cut_points[b]
                                      : std::numeric_limits<bst_float>::max();

                bst_float gain = split_gain(GL, HL, GR, HR);
                if (gain > local_best.gain) {
                    local_best = { gain, feat, split_val,
                                   (G_total - GL > 0.0), true };
                }
            }

            int tid = omp_get_thread_num();
            if (local_best.gain > thread_bests[tid].gain)
                thread_bests[tid] = local_best;
        }
    }
    for (const auto& tb : thread_bests)
        if (tb.gain > best.gain) best = tb;

#else
    for (int ci = 0; ci < n_cols; ++ci) {
        bst_uint feat = col_indices[ci];
        if (feat >= static_cast<bst_uint>(hists.size())) continue;
        const FeatureHistogram& hist = hists[feat];
        const bst_uint n_bins = hist.n_bins();

        double GL = 0.0, HL = 0.0;
        for (bst_uint b = 0; b + 1 < n_bins; ++b) {
            GL += hist.bins[b].sum_grad;
            HL += hist.bins[b].sum_hess;
            double GR = G_total - GL, HR = H_total - HL;

            if (HL < cfg_.min_child_weight) continue;
            if (HR < cfg_.min_child_weight) continue;

            bst_float split_val = (b < static_cast<bst_uint>(hist.cut_points.size()))
                                  ? hist.cut_points[b]
                                  : std::numeric_limits<bst_float>::max();

            bst_float gain = split_gain(GL, HL, GR, HR);
            if (gain > best.gain) {
                best = { gain, feat, split_val, (G_total - GL > 0.0), true };
            }
        }
    }
#endif

    return best;
}

} // namespace xgb
