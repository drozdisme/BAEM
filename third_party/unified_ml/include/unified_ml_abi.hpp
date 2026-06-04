#pragma once
// ══════════════════════════════════════════════════════════════════════════════
//  unified_ml — ABI / symbol visibility control
//
//  ABI NOTE
//
//  Shared-library packaging follows conventional major-version SONAME rules.
//  A matching major version is treated as the minimum compatibility boundary,
//  but binary compatibility across all minor releases is best-effort, not an
//  unconditional guarantee.
//
//  SYMBOL VISIBILITY MACROS
//     
//  UNIFIED_ML_API  — marks a symbol as part of the public ABI.
//        Used on class declarations, free functions, etc.
//  UNIFIED_ML_LOCAL — explicitly marks a symbol as hidden (not exported),
//        useful for internal helpers in headers.
//
//  USAGE IN HEADERS
//     
//  class UNIFIED_ML_API MyClass { ... };
//  UNIFIED_ML_API void my_function();
//
//  BUILD CONFIGURATION
//    
//  Shared build (CMake sets -DUNIFIED_ML_EXPORTS):
//  UNIFIED_ML_API = __declspec(dllexport) on MSVC / __attribute__((visibility("default"))) on GCC
//  Consumer of shared build (CMake sets -DUNIFIED_ML_DLL):
//  UNIFIED_ML_API = __declspec(dllimport) on MSVC / default visibility on GCC
//  Static build (neither flag set):
//  UNIFIED_ML_API = empty (no decoration needed)
// ══════════════════════════════════════════════════════════════════════════════

// Compiler-specific visibility        
#if defined(_WIN32) || defined(__CYGWIN__)
  // Windows: DLL import/export
  #if defined(UNIFIED_ML_EXPORTS)
    // Building the DLL — export symbols
    #if defined(_MSC_VER)
    #define UNIFIED_ML_API __declspec(dllexport)
    #else
    #define UNIFIED_ML_API __attribute__((dllexport))
    #endif
  #elif defined(UNIFIED_ML_DLL)
    // Consuming the DLL — import symbols
    #if defined(_MSC_VER)
    #define UNIFIED_ML_API __declspec(dllimport)
    #else
    #define UNIFIED_ML_API __attribute__((dllimport))
    #endif
  #else
    // Static build — no decoration
    #define UNIFIED_ML_API
  #endif
  #define UNIFIED_ML_LOCAL // no hidden on Windows
#else
  // GCC / Clang on Linux / macOS
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define UNIFIED_ML_API __attribute__((visibility("default")))
    #define UNIFIED_ML_LOCAL __attribute__((visibility("hidden")))
  #else
    #define UNIFIED_ML_API
    #define UNIFIED_ML_LOCAL
  #endif
#endif

// Deprecation           
// Mark functions deprecated while keeping them in the ABI for one major cycle:
// UNIFIED_ML_DEPRECATED("Use new_func() instead")
// void old_func();
#if defined(_MSC_VER)
  #define UNIFIED_ML_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__GNUC__) || defined(__clang__)
  #define UNIFIED_ML_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
  #define UNIFIED_ML_DEPRECATED(msg)
#endif

// Inline hint           
#if defined(_MSC_VER)
  #define UNIFIED_ML_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define UNIFIED_ML_FORCEINLINE __attribute__((always_inline)) inline
#else
  #define UNIFIED_ML_FORCEINLINE inline
#endif

// ABI version check          
// Used in shared-lib consumers to verify ABI compatibility at runtime:
// unified_ml_check_abi(); // throws if mismatch
#include <cstdio>   // std::sscanf
#include <cstring>  // std::memcpy
#include <stdexcept>  // std::runtime_error
#include <string>   // std::string
#include "unified_ml_version.hpp"

namespace unified_ml {
/// Returns the ABI version string baked into the shared library.
/// Compare against UNIFIED_ML_VERSION_STRING from headers to detect mismatch.
UNIFIED_ML_API const char* abi_version() noexcept;

/// Throws std::runtime_error if header and library major versions differ.
/// This is a coarse compatibility check, not proof of full ABI identity.
inline void check_abi() {
  const char* lib_ver = abi_version();
  if (!lib_ver || lib_ver[0] == '\0') return;
  // Parse major version: read digits up to first '.'
  // Using only <cstdio>/<cstring> — avoids std::sscanf portability issues
  // and MSVC sscanf deprecation warnings.
  int lib_major = 0;
  for (const char* p = lib_ver; *p && *p != '.'; ++p) {
    if (*p >= '0' && *p <= '9')
    lib_major = lib_major * 10 + (*p - '0');
  }
  if (lib_major != UNIFIED_ML_VERSION_MAJOR) {
    throw std::runtime_error(
    std::string("unified_ml ABI mismatch: headers=")
    + UNIFIED_ML_VERSION_STRING + " library=" + lib_ver
    + " (major version must match)");
  }
}
} // namespace unified_ml
