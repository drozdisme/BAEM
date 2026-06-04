// compat.hpp — Cross-platform compatibility macros for MSVC / GCC / Clang.
//
// Fixes three MSVC issues:
//   1. __restrict__ → __restrict (MSVC keyword)
//   2. std::aligned_alloc → _aligned_malloc/_aligned_free
//   3. #pragma omp simd → requires /openmp:experimental on MSVC

#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

//   restrict qualifier                             
// MSVC uses __restrict, GCC/Clang use __restrict__
#ifdef _MSC_VER
  #define HPC_RESTRICT __restrict
#else
  #define HPC_RESTRICT __restrict__
#endif

//   Aligned allocation                            
// MSVC lacks std::aligned_alloc (C11). Use _aligned_malloc/_aligned_free.
// GCC/Clang have std::aligned_alloc.
inline void* hpc_aligned_alloc(std::size_t alignment, std::size_t size) {
    if (size == 0) return nullptr;
#ifdef _MSC_VER
    void* p = _aligned_malloc(size, alignment);
#else
    // aligned_alloc requires size to be a multiple of alignment
    std::size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, aligned_size);
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

inline void hpc_aligned_free(void* p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}

//   SIMD pragma                                
// MSVC requires /openmp:experimental for #pragma omp simd.
// If not available, we just skip the pragma (code still works, just no hint).
// Usage: HPC_PRAGMA_OMP_SIMD before a for loop.
#if defined(_MSC_VER) && !defined(_OPENMP_SIMD)
  // MSVC without /openmp:experimental — skip simd pragma
  #define HPC_PRAGMA_OMP_SIMD
  #define HPC_PRAGMA_OMP_SIMD_REDUCTION(op, var)
#elif defined(_OPENMP)
  #define HPC_PRAGMA_OMP_SIMD _Pragma("omp simd")
  #define HPC_PRAGMA_OMP_SIMD_REDUCTION(op, var) _Pragma("omp simd reduction(" #op ":" #var ")")
#else
  #define HPC_PRAGMA_OMP_SIMD
  #define HPC_PRAGMA_OMP_SIMD_REDUCTION(op, var)
#endif
