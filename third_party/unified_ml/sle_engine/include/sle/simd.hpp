#pragma once

#include <cstddef>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
  #include <immintrin.h>
  #define SLE_X86_SIMD 1
#else
  #define SLE_X86_SIMD 0
#endif

#if SLE_X86_SIMD
  #define SLE_HAS_AVX512 1
#else
  #define SLE_HAS_AVX512 0
#endif

namespace sle::simd {

inline bool runtime_has_avx512() noexcept {
#if !SLE_HAS_AVX512
    return false;
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
    static const bool cached = __builtin_cpu_supports("avx512f");
    return cached;
#else
    return false;
#endif
#elif defined(_MSC_VER)
    int cpu_info[4] = {0, 0, 0, 0};
    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 16)) != 0;
#else
    return false;
#endif
}

inline constexpr std::size_t register_bytes = 64;

} // namespace sle::simd
