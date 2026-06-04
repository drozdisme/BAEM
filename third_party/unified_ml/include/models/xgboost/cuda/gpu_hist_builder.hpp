#pragma once
//                                        
//  GPU Histogram Builder  (P4-5)
//
//  Runtime device selection:
//    HistDevice::CPU  — uses existing HistogramBuilder (OpenMP)
//    HistDevice::GPU  — uses CUDA kernel (if compiled with -DXGB_USE_CUDA)
//
//  CUDA kernel design:
//    • One CUDA thread per sample
//    • Shared-memory partial histograms per block
//    • Atomic adds + block reduction → global histogram
//    • Fallback to CPU if no CUDA device found at runtime
//                                        
#pragma once
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include <string>
#include <vector>

namespace xgb {

enum class HistDevice { CPU, GPU };

// Parse device string: "cpu_hist" / "gpu_hist"
inline HistDevice parse_hist_device(const std::string& s) {
    if (s == "gpu_hist") return HistDevice::GPU;
    return HistDevice::CPU;
}

//   Unified histogram builder (routes at runtime)                
class UnifiedHistBuilder {
public:
    explicit UnifiedHistBuilder(HistDevice device = HistDevice::CPU,
                                int n_bins = 256);

    // Build histograms for all active features
    // Returns vector<FeatureHistogram> identical to HistogramBuilder output
    std::vector<FeatureHistogram> build(
        const std::vector<GradientPair>&  grads,
        const std::vector<bst_uint>&      row_indices,
        const std::vector<bst_float>&     feature_values,
        const std::vector<bst_float>&     cut_points,
        bst_uint                          n_features);

    HistDevice device() const { return device_; }
    bool       cuda_available() const;

private:
    HistDevice device_;
    int        n_bins_;
    // CPU fallback implemented inline — no embedded HistogramBuilder needed

#ifdef XGB_USE_CUDA
    void build_gpu(
        const std::vector<GradientPair>&  grads,
        const std::vector<bst_uint>&      row_indices,
        const float*                       d_features,
        const float*                       d_cuts,
        std::vector<FeatureHistogram>&     out,
        bst_uint                           n_features,
        bst_uint                           n_samples);
#endif
};

} // namespace xgb
