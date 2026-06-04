// src/cuda/gpu_hist_builder.cpp  — P4-5 GPU histogram, CPU fallback
#include "models/xgboost/cuda/gpu_hist_builder.hpp"
#include <algorithm>
#include <stdexcept>

namespace xgb {

UnifiedHistBuilder::UnifiedHistBuilder(HistDevice device, int n_bins)
    : device_(device), n_bins_(n_bins)
{
#ifndef XGB_USE_CUDA
    if (device == HistDevice::GPU)
        device_ = HistDevice::CPU;  // silent fallback
#else
    if (device == HistDevice::GPU && !cuda_available())
        device_ = HistDevice::CPU;
#endif
}

bool UnifiedHistBuilder::cuda_available() const {
#ifdef XGB_USE_CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    return count > 0;
#else
    return false;
#endif
}

//   CPU fallback: build histograms from flat feature_values + cut_points    
// feature_values layout: row-major [n_samples × n_features]
// cut_points: shared across all features (uniform binning proxy)
std::vector<FeatureHistogram> UnifiedHistBuilder::build(
    const std::vector<GradientPair>&  grads,
    const std::vector<bst_uint>&      row_indices,
    const std::vector<bst_float>&     feature_values,
    const std::vector<bst_float>&     cut_points,
    bst_uint                          n_features)
{
#ifdef XGB_USE_CUDA
    if (device_ == HistDevice::GPU) {
        // GPU path: delegate to CUDA kernel (compiled in gpu_hist_kernel.cu)
        // For now fall through to CPU until kernel is linked
    }
#endif
    //   CPU path                               
    bst_uint n_bins = cut_points.empty() ? 1u
                    : static_cast<bst_uint>(cut_points.size());

    // Build one FeatureHistogram per feature
    std::vector<FeatureHistogram> hists(n_features);
    for (bst_uint f = 0; f < n_features; ++f) {
        hists[f].bins.resize(n_bins + 1);   // +1 for values above last cut
        hists[f].cut_points = cut_points;
    }

    for (bst_uint ri : row_indices) {
        const GradientPair& gp = grads[ri];
        for (bst_uint f = 0; f < n_features; ++f) {
            bst_float val = feature_values[ri * n_features + f];
            // Binary search: find bin index
            auto it = std::upper_bound(cut_points.begin(), cut_points.end(), val);
            bst_uint bin = static_cast<bst_uint>(it - cut_points.begin());
            if (bin >= static_cast<bst_uint>(hists[f].bins.size()))
                bin  = static_cast<bst_uint>(hists[f].bins.size()) - 1u;
            hists[f].bins[bin].add(gp);
        }
    }
    return hists;
}

} // namespace xgb
