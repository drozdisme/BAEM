//                                        
//  gpu_hist_kernel.cu   — CUDA histogram construction (P4-5)
//
//  Design:
//    • gridDim.x  = number of features
//    • blockDim.x = BLOCK_SIZE threads
//    • Each block owns a shared-memory partial histogram for its feature
//    • Threads iterate over samples: one thread per sample per feature-pass
//    • Atomic adds into shared memory, then block reduction to global memory
//                                        
#ifdef XGB_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

//   Gradient pair in GPU-friendly layout (SoA)                 
struct GradPairGPU { float grad, hess; };

//   Histogram bin                                
struct BinGPU { float sum_grad, sum_hess; };

//                                        
//  Kernel: build_histograms_kernel
//
//  Parameters:
//    grads       — gradient pairs (n_samples)
//    row_indices — active row indices (n_active)
//    features    — feature matrix [row * n_features + col] (n_samples * n_features)
//    cuts        — cut points for this feature (n_bins per feature)
//    n_cuts      — number of cut points per feature
//    n_active    — number of active samples
//    n_features  — total feature count
//    hist        — output histograms [feature * n_bins + bin] (n_features * n_bins)
//                                        
#define BLOCK_SIZE 256
#define MAX_BINS   256

__global__ void build_histograms_kernel(
    const GradPairGPU* __restrict__ grads,
    const uint32_t*   __restrict__ row_indices,
    const float*      __restrict__ features,
    const float*      __restrict__ cuts,        // [n_features * MAX_BINS]
    const uint32_t*   __restrict__ n_cuts,      // [n_features]
    uint32_t                       n_active,
    uint32_t                       n_features,
    float*                         hist_grad,   // [n_features * MAX_BINS]
    float*                         hist_hess)
{
    // Each block processes one feature
    uint32_t feat_id = blockIdx.x;
    if (feat_id >= n_features) return;

    // Shared memory histogram for this feature's bins
    extern __shared__ float smem[];
    float* s_grad = smem;
    float* s_hess = smem + MAX_BINS;

    // Zero shared memory
    for (uint32_t b = threadIdx.x; b < MAX_BINS; b += blockDim.x) {
        s_grad[b] = 0.f;
        s_hess[b] = 0.f;
    }
    __syncthreads();

    // Each thread processes multiple samples
    uint32_t nc = n_cuts[feat_id];

    for (uint32_t i = threadIdx.x; i < n_active; i += blockDim.x) {
        uint32_t row   = row_indices[i];
        float    fval  = features[row * n_features + feat_id];
        const float* c = cuts + feat_id * MAX_BINS;

        // Binary search for bin index
        uint32_t bin = nc - 1;
        for (uint32_t k = 0; k < nc; ++k) {
            if (fval < c[k]) { bin = k; break; }
        }

        atomicAdd(&s_grad[bin], grads[row].grad);
        atomicAdd(&s_hess[bin], grads[row].hess);
    }
    __syncthreads();

    // Write shared → global
    float* g_grad = hist_grad + feat_id * MAX_BINS;
    float* g_hess = hist_hess + feat_id * MAX_BINS;
    for (uint32_t b = threadIdx.x; b < MAX_BINS; b += blockDim.x) {
        atomicAdd(&g_grad[b], s_grad[b]);
        atomicAdd(&g_hess[b], s_hess[b]);
    }
}

//   Host launcher (called from gpu_hist_builder.cpp when CUDA enabled)     
extern "C"
void launch_hist_kernel(
    const GradPairGPU* d_grads,
    const uint32_t*    d_row_indices,
    const float*       d_features,
    const float*       d_cuts,
    const uint32_t*    d_n_cuts,
    uint32_t           n_active,
    uint32_t           n_features,
    uint32_t           n_bins,
    float*             d_hist_grad,
    float*             d_hist_hess)
{
    dim3 grid(n_features);
    dim3 block(BLOCK_SIZE);
    size_t smem = 2 * n_bins * sizeof(float);

    build_histograms_kernel<<<grid, block, smem>>>(
        d_grads, d_row_indices, d_features,
        d_cuts, d_n_cuts,
        n_active, n_features,
        d_hist_grad, d_hist_hess);
    cudaDeviceSynchronize();
}

#endif // XGB_USE_CUDA
