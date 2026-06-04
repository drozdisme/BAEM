// hpc_kernels.hpp — Production HPC CPU kernels for unified_ml.
//
// PORTABLE: Compiles on GCC, Clang, MSVC. Auto-detects AVX-512.
//   - With AVX-512: register-blocked 6x16 micro-kernel, 80+ GFLOP/s
//   - Without AVX-512: auto-vectorized fallbacks via #pragma omp simd
//
// Zero external dependencies. Only: <immintrin.h>, <cstring>, <cmath>, <omp.h>

#pragma once

#include "core/compat.hpp"
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <algorithm>

//   AVX-512 detection                             
#if defined(__AVX512F__)
  #define HPC_HAS_AVX512 1
#else
  #define HPC_HAS_AVX512 0
#endif

// Only include x86 SIMD intrinsics on x86/x64 architectures
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&     (HPC_HAS_AVX512 || defined(__AVX2__) || defined(__AVX__) || defined(_MSC_VER))
#  include <immintrin.h>
#endif

// Portable horizontal sum (some compilers lack _mm512_reduce_add_pd as builtin)
#if HPC_HAS_AVX512
static inline double hpc_reduce_add(__m512d v) {
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d s  = _mm256_add_pd(lo, hi);
    __m128d s2 = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
    return _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
}
#endif

#ifdef _OPENMP
  #include <omp.h>
#else
  static inline int omp_in_parallel()     { return 0; }
  static inline int omp_get_max_threads() { return 1; }
#endif

//   Constants                                 
#ifndef HPC_CACHE_LINE
#define HPC_CACHE_LINE 64
#endif
#ifndef HPC_MR
#define HPC_MR  6
#endif
#ifndef HPC_NR
#define HPC_NR  16
#endif
#ifndef HPC_MC
#define HPC_MC  96
#endif
#ifndef HPC_NC
#define HPC_NC  256
#endif
#ifndef HPC_KC
#define HPC_KC  256
#endif
#ifndef HPC_PREFETCH_DIST
#define HPC_PREFETCH_DIST 64
#endif
#ifndef HPC_PAR_THRESHOLD
#define HPC_PAR_THRESHOLD 100000
#endif

namespace hpc {

inline bool runtime_has_avx512() noexcept {
#if !HPC_HAS_AVX512
    return false;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool cached = []() noexcept {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
        return __builtin_cpu_supports("avx512f");
#else
        return false;
#endif
#elif defined(_MSC_VER)
        int cpu_info[4] = {0, 0, 0, 0};
        __cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 16)) != 0; // AVX-512F
#else
        return false;
#endif
    }();
    return cached;
#else
    return false;
#endif
}

//   Aligned alloc                               
inline double* alloc64(std::size_t n) {
    if (n == 0) return nullptr;
    std::size_t bytes = ((n * sizeof(double) + 63) / 64) * 64;
    return static_cast<double*>(hpc_aligned_alloc(64, bytes));
}
inline void free64(double* p) {
    hpc_aligned_free(p);
}

inline double* thread_local_scratch(std::size_t required_elems) {
    struct ScratchBuffer {
        double* ptr = nullptr;
        std::size_t capacity = 0;
        ~ScratchBuffer() { free64(ptr); }
    };
    thread_local ScratchBuffer scratch;
    if (scratch.capacity < required_elems) {
        free64(scratch.ptr);
        scratch.ptr = alloc64(required_elems);
        scratch.capacity = required_elems;
    }
    return scratch.ptr;
}

//   Padded accumulator (false-sharing elimination)               
struct alignas(HPC_CACHE_LINE) PaddedAccum {
    double value;
    char _pad[HPC_CACHE_LINE - sizeof(double)];
    PaddedAccum() : value(0.0) { std::memset(_pad, 0, sizeof(_pad)); }
};

// ═════════════════════════════════════════════════════════════════════════════
// §A  PORTABLE GEMV DOT PRODUCT (used everywhere below)
// ═════════════════════════════════════════════════════════════════════════════
// Returns dot(W_row, x) using best available SIMD.
static inline double dot_product(const double* HPC_RESTRICT w,
                                  const double* HPC_RESTRICT x,
                                  std::size_t n) noexcept {
#if HPC_HAS_AVX512
    if (runtime_has_avx512()) {
    __m512d sv = _mm512_setzero_pd();
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        if (j + 72 <= n)
            _mm_prefetch((const char*)(w + j + 64), _MM_HINT_T0);
        sv = _mm512_fmadd_pd(_mm512_loadu_pd(w + j), _mm512_loadu_pd(x + j), sv);
    }
    double acc = hpc_reduce_add(sv);
    for (; j < n; ++j) acc += w[j] * x[j];
    return acc;
    }
#endif
    double acc_scalar = 0.0;
    #pragma omp simd reduction(+:acc_scalar)
    for (std::size_t j = 0; j < n; ++j) acc_scalar += w[j] * x[j];
    return acc_scalar;
}
// ═════════════════════════════════════════════════════════════════════════════
// §4  MICRO-KERNELS
// ═════════════════════════════════════════════════════════════════════════════

#if HPC_HAS_AVX512
// 6x16 register-blocked micro-kernel: 12 ZMM accumulators, zero spills.
inline void microkernel_6x16_packed(
    const double* HPC_RESTRICT A_packed, const double* HPC_RESTRICT B_packed,
    double* HPC_RESTRICT C, std::size_t K, std::size_t ldc
) noexcept {
    __m512d c00=_mm512_setzero_pd(),c01=_mm512_setzero_pd();
    __m512d c10=_mm512_setzero_pd(),c11=_mm512_setzero_pd();
    __m512d c20=_mm512_setzero_pd(),c21=_mm512_setzero_pd();
    __m512d c30=_mm512_setzero_pd(),c31=_mm512_setzero_pd();
    __m512d c40=_mm512_setzero_pd(),c41=_mm512_setzero_pd();
    __m512d c50=_mm512_setzero_pd(),c51=_mm512_setzero_pd();
    for (std::size_t k = 0; k < K; ++k) {
        const double* ap = A_packed + k * HPC_MR;
        const double* bp = B_packed + k * HPC_NR;
        if (k + HPC_PREFETCH_DIST < K) {
            _mm_prefetch((const char*)(B_packed+(k+HPC_PREFETCH_DIST)*HPC_NR), _MM_HINT_T0);
            _mm_prefetch((const char*)(B_packed+(k+HPC_PREFETCH_DIST)*HPC_NR+8), _MM_HINT_T0);
        }
        __m512d b0=_mm512_load_pd(bp), b1=_mm512_load_pd(bp+8), a;
        a=_mm512_set1_pd(ap[0]); c00=_mm512_fmadd_pd(a,b0,c00); c01=_mm512_fmadd_pd(a,b1,c01);
        a=_mm512_set1_pd(ap[1]); c10=_mm512_fmadd_pd(a,b0,c10); c11=_mm512_fmadd_pd(a,b1,c11);
        a=_mm512_set1_pd(ap[2]); c20=_mm512_fmadd_pd(a,b0,c20); c21=_mm512_fmadd_pd(a,b1,c21);
        a=_mm512_set1_pd(ap[3]); c30=_mm512_fmadd_pd(a,b0,c30); c31=_mm512_fmadd_pd(a,b1,c31);
        a=_mm512_set1_pd(ap[4]); c40=_mm512_fmadd_pd(a,b0,c40); c41=_mm512_fmadd_pd(a,b1,c41);
        a=_mm512_set1_pd(ap[5]); c50=_mm512_fmadd_pd(a,b0,c50); c51=_mm512_fmadd_pd(a,b1,c51);
    }
    #define SR(r,c0r,c1r) \
        _mm512_storeu_pd(C+(r)*ldc,   _mm512_add_pd(_mm512_loadu_pd(C+(r)*ldc),  c0r)); \
        _mm512_storeu_pd(C+(r)*ldc+8, _mm512_add_pd(_mm512_loadu_pd(C+(r)*ldc+8),c1r));
    SR(0,c00,c01);SR(1,c10,c11);SR(2,c20,c21);SR(3,c30,c31);SR(4,c40,c41);SR(5,c50,c51);
    #undef SR
}
#endif // HPC_HAS_AVX512

// Edge kernel: works on any platform (auto-vectorized via omp simd or AVX-512)
inline void microkernel_edge(
    const double* HPC_RESTRICT A, const double* HPC_RESTRICT B, double* HPC_RESTRICT C,
    std::size_t M_blk, std::size_t N_blk, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) noexcept {
#if HPC_HAS_AVX512
    const bool use_avx512 = runtime_has_avx512();
#endif
    for (std::size_t i = 0; i < M_blk; ++i)
        for (std::size_t k = 0; k < K; ++k) {
            double a_ik = A[i*lda+k]; if (a_ik == 0.0) continue;
            const double* Bk = B + k*ldb; double* Ci = C + i*ldc;
#if HPC_HAS_AVX512
            if (use_avx512) {
                __m512d av = _mm512_set1_pd(a_ik);
                std::size_t j = 0;
                for (; j+8<=N_blk; j+=8)
                    _mm512_storeu_pd(Ci+j, _mm512_fmadd_pd(av, _mm512_loadu_pd(Bk+j), _mm512_loadu_pd(Ci+j)));
                for (; j<N_blk; ++j) Ci[j] += a_ik*Bk[j];
            } else {
                #pragma omp simd
                for (std::size_t j = 0; j < N_blk; ++j) Ci[j] += a_ik*Bk[j];
            }
#else
            #pragma omp simd
            for (std::size_t j = 0; j < N_blk; ++j) Ci[j] += a_ik*Bk[j];
#endif
        }
}

//   Pack routines                               
inline void pack_B_panel(const double* HPC_RESTRICT B, double* HPC_RESTRICT dst,
    std::size_t K_blk, std::size_t N_blk, std::size_t ldb) noexcept {
#if HPC_HAS_AVX512
    const bool use_avx512 = runtime_has_avx512();
#endif
    std::size_t nf=N_blk/HPC_NR, nr=N_blk%HPC_NR;
    for (std::size_t jp=0;jp<nf;++jp) { std::size_t jj=jp*HPC_NR;
        for (std::size_t k=0;k<K_blk;++k) { const double* s=B+k*ldb+jj;
#if HPC_HAS_AVX512
            if (use_avx512) {
                _mm512_store_pd(dst,   _mm512_loadu_pd(s));
                _mm512_store_pd(dst+8, _mm512_loadu_pd(s+8));
            } else {
                std::memcpy(dst, s, HPC_NR*sizeof(double));
            }
#else
            std::memcpy(dst, s, HPC_NR*sizeof(double));
#endif
            dst+=HPC_NR; }}
    if (nr>0) { std::size_t jj=nf*HPC_NR;
        for (std::size_t k=0;k<K_blk;++k) { const double* s=B+k*ldb+jj;
            for (std::size_t jr=0;jr<HPC_NR;++jr) *dst++=(jr<nr)?s[jr]:0.0; }}
}
inline void pack_A_panel(const double* HPC_RESTRICT A, double* HPC_RESTRICT dst,
    std::size_t M_blk, std::size_t K_blk, std::size_t lda) noexcept {
    std::size_t mf=M_blk/HPC_MR, mr=M_blk%HPC_MR;
    for (std::size_t ip=0;ip<mf;++ip) { std::size_t ii=ip*HPC_MR;
        for (std::size_t k=0;k<K_blk;++k) for (std::size_t ir=0;ir<HPC_MR;++ir) *dst++=A[(ii+ir)*lda+k]; }
    if (mr>0) { std::size_t ii=mf*HPC_MR;
        for (std::size_t k=0;k<K_blk;++k) for (std::size_t ir=0;ir<HPC_MR;++ir)
            *dst++=(ir<mr)?A[(ii+ir)*lda+k]:0.0; }
}

// ═════════════════════════════════════════════════════════════════════════════
// §7  PARALLEL GEMM: C[M×N] += A[M×K] × B[K×N]
// ═════════════════════════════════════════════════════════════════════════════
inline void gemm_hpc(
    const double* HPC_RESTRICT A, const double* HPC_RESTRICT B, double* HPC_RESTRICT C,
    std::size_t M, std::size_t K, std::size_t N,
    std::size_t lda, std::size_t ldb, std::size_t ldc
) noexcept {
    if (M*N*K < 512) { microkernel_edge(A,B,C,M,N,K,lda,ldb,ldc); return; }

#if HPC_HAS_AVX512
    const bool use_avx512 = runtime_has_avx512();
#endif
    double* B_packed = thread_local_scratch(HPC_KC * HPC_NC);
    const bool can_par = (M > (std::size_t)HPC_MR*2)
                         && !omp_in_parallel()
                         && (M*K*N > (std::size_t)HPC_PAR_THRESHOLD);

    for (std::size_t k0=0;k0<K;k0+=HPC_KC) {
        std::size_t kb=std::min((std::size_t)HPC_KC,K-k0);
        for (std::size_t j0=0;j0<N;j0+=HPC_NC) {
            std::size_t nb=std::min((std::size_t)HPC_NC,N-j0);
            pack_B_panel(B+k0*ldb+j0, B_packed, kb, nb, ldb);
            std::size_t nmc=(M+HPC_MC-1)/HPC_MC;

            #pragma omp parallel for schedule(static) if(can_par)
            for (int bi=0;bi<(int)nmc;++bi) {
                std::size_t i0=bi*HPC_MC, mb=std::min((std::size_t)HPC_MC,M-i0);
                alignas(64) double A_local[HPC_MC*HPC_KC];
                pack_A_panel(A+i0*lda+k0, A_local, mb, kb, lda);
                std::size_t mp=(mb+HPC_MR-1)/HPC_MR, nfj=nb/HPC_NR, nrj=nb%HPC_NR;
                for (std::size_t ip=0;ip<mp;++ip) {
                    std::size_t ii=i0+ip*HPC_MR, mra=std::min((std::size_t)HPC_MR,M-ii);
                    for (std::size_t jp=0;jp<nfj;++jp) {
                        std::size_t jj=j0+jp*HPC_NR;
#if HPC_HAS_AVX512
                        if (use_avx512 && mra==HPC_MR)
                            microkernel_6x16_packed(A_local+ip*kb*HPC_MR,B_packed+jp*kb*HPC_NR,C+ii*ldc+jj,kb,ldc);
                        else
#endif
                            microkernel_edge(A+ii*lda+k0,B+k0*ldb+jj,C+ii*ldc+jj,mra,HPC_NR,kb,lda,ldb,ldc);
                    }
                    if (nrj>0) microkernel_edge(A+ii*lda+k0,B+k0*ldb+j0+nfj*HPC_NR,
                                                 C+ii*ldc+j0+nfj*HPC_NR,mra,nrj,kb,lda,ldb,ldc);
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §8  FUSED GEMV + BIAS + ACTIVATION (LEAF — ZERO OMP)
// ═════════════════════════════════════════════════════════════════════════════
// Two-phase: Phase 1 (GEMV+bias) → Phase 2 (vectorized activation).
enum class FusedAct { IDENTITY, TANH, RELU };

inline void fused_gemv_bias_act(
    const double* HPC_RESTRICT W, const double* HPC_RESTRICT x,
    const double* HPC_RESTRICT bias, double* HPC_RESTRICT out,
    std::size_t N_out, std::size_t N_in, FusedAct act
) noexcept {
    // Phase 1: z[i] = dot(W[i,:], x) + bias[i]
    for (std::size_t i = 0; i < N_out; ++i) {
        out[i] = dot_product(W + i * N_in, x, N_in) + bias[i];
    }
    // Phase 2: vectorized activation (separate loop — GCC auto-vectorizes tanh)
    switch (act) {
        case FusedAct::TANH:
            #pragma omp simd
            for (std::size_t i=0;i<N_out;++i) out[i]=std::tanh(out[i]);
            break;
        case FusedAct::RELU:
            #pragma omp simd
            for (std::size_t i=0;i<N_out;++i) out[i]=out[i]>0.0?out[i]:0.0;
            break;
        case FusedAct::IDENTITY: break;
    }
}

// Batched: parallel over batch with adaptive threshold + anti-nesting.
inline void fused_batch_forward(
    const double* HPC_RESTRICT W, const double* HPC_RESTRICT bias,
    const double* HPC_RESTRICT X_batch, double* HPC_RESTRICT Out_batch,
    std::size_t batch, std::size_t N_out, std::size_t N_in, FusedAct act
) noexcept {
    const bool par = (batch*N_out*N_in > (std::size_t)HPC_PAR_THRESHOLD)
                     && !omp_in_parallel() && (omp_get_max_threads()>1);
    if (par) {
        #pragma omp parallel for schedule(static)
        for (int b=0;b<(int)batch;++b)
            fused_gemv_bias_act(W,X_batch+b*N_in,bias,Out_batch+b*N_out,N_out,N_in,act);
    } else {
        for (std::size_t b=0;b<batch;++b)
            fused_gemv_bias_act(W,X_batch+b*N_in,bias,Out_batch+b*N_out,N_out,N_in,act);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §9  PINN FORWARD KERNELS (TWO-PHASE)
// ═════════════════════════════════════════════════════════════════════════════

// §9a INFERENCE-ONLY (no derivative caching)
inline void fused_pinn_forward_inference(
    const double* HPC_RESTRICT W, const double* HPC_RESTRICT bias,
    const double* HPC_RESTRICT a_prev,
    double* HPC_RESTRICT z_out, double* HPC_RESTRICT a_out,
    std::size_t N_out, std::size_t N_in, bool is_last
) noexcept {
    for (std::size_t i=0;i<N_out;++i)
        z_out[i] = dot_product(W + i*N_in, a_prev, N_in) + bias[i];
    if (!is_last) {
        #pragma omp simd
        for (std::size_t i=0;i<N_out;++i) a_out[i]=std::tanh(z_out[i]);
    } else {
        std::memcpy(a_out, z_out, N_out*sizeof(double));
    }
}

// §9b TRAINING (+ s/p/q derivative caching)
inline void fused_pinn_forward(
    const double* HPC_RESTRICT W, const double* HPC_RESTRICT bias,
    const double* HPC_RESTRICT a_prev,
    double* HPC_RESTRICT z_out, double* HPC_RESTRICT a_out,
    double* HPC_RESTRICT s_out, double* HPC_RESTRICT p_out, double* HPC_RESTRICT q_out,
    std::size_t N_out, std::size_t N_in, bool is_last
) noexcept {
    for (std::size_t i=0;i<N_out;++i)
        z_out[i] = dot_product(W + i*N_in, a_prev, N_in) + bias[i];
    if (!is_last) {
        #pragma omp simd
        for (std::size_t i=0;i<N_out;++i) a_out[i]=std::tanh(z_out[i]);
        #pragma omp simd
        for (std::size_t i=0;i<N_out;++i) {
            double a=a_out[i], a2=a*a, s=1.0-a2;
            s_out[i]=s; p_out[i]=-2.0*a*s; q_out[i]=(-2.0+6.0*a2)*s;
        }
    } else {
        std::memcpy(a_out, z_out, N_out*sizeof(double));
        for (std::size_t i=0;i<N_out;++i) { s_out[i]=1.0; p_out[i]=0.0; q_out[i]=0.0; }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// §10-12  TRANSPOSE, DOT, GEMV, GEMVT, OUTER
// ═════════════════════════════════════════════════════════════════════════════

inline double parallel_dot(const double* HPC_RESTRICT a, const double* HPC_RESTRICT b, std::size_t n) noexcept {
    double total=0.0;
    #pragma omp parallel reduction(+:total) if(n>100000 && !omp_in_parallel())
    { double loc=0.0;
      #pragma omp for schedule(static) nowait
      for (int i=0;i<(int)n;++i) loc+=a[i]*b[i];
      total+=loc; }
    return total;
}

inline void transpose_hpc(const double* HPC_RESTRICT src, double* HPC_RESTRICT dst,
    std::size_t rows, std::size_t cols) noexcept {
    constexpr std::size_t TB=32;
    const bool par=(rows*cols>10000)&&!omp_in_parallel();
    #pragma omp parallel for schedule(static) if(par)
    for (int i0=0;i0<(int)rows;i0+=TB) {
        for (std::size_t j0=0;j0<cols;j0+=TB) {
        std::size_t ie=std::min((std::size_t)i0+TB,rows), je=std::min(j0+TB,cols);
        for (std::size_t i=i0;i<ie;++i) for (std::size_t j=j0;j<je;++j)
            dst[j*rows+i]=src[i*cols+j]; }}
}

// PINN GEMV/GEMVT/OUTER — use dot_product for portability
inline void gemv_hpc(const double* HPC_RESTRICT W, const double* HPC_RESTRICT x,
    double* HPC_RESTRICT y, std::size_t n, std::size_t m) noexcept {
    for (std::size_t i=0;i<n;++i)
        y[i] += dot_product(W + i*m, x, m);
}

inline void gemvT_hpc(const double* HPC_RESTRICT W, const double* HPC_RESTRICT x,
    double* HPC_RESTRICT y, std::size_t n, std::size_t m) noexcept {
    for (std::size_t i=0;i<n;++i) {
        const double xi = x[i]; const double* Wi = W + i*m;
        #pragma omp simd
        for (std::size_t j=0;j<m;++j) y[j]+=xi*Wi[j];
    }
}

inline void outer_hpc(const double* HPC_RESTRICT u, const double* HPC_RESTRICT v,
    double* HPC_RESTRICT M, std::size_t n, std::size_t m) noexcept {
    for (std::size_t i=0;i<n;++i) {
        double ui=u[i]; double* Mi=M+i*m;
        #pragma omp simd
        for (std::size_t j=0;j<m;++j) Mi[j]+=ui*v[j];
    }
}

} // namespace hpc
