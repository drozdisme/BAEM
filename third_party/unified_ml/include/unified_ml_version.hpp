#pragma once

// unified_ml version macros          
// Use these to write version-guarded code:
//
// #if UNIFIED_ML_VERSION >= UNIFIED_ML_MAKE_VERSION(1, 1, 0)
//   // use new API
// #endif

#define UNIFIED_ML_VERSION_MAJOR 1
#define UNIFIED_ML_VERSION_MINOR 0
#define UNIFIED_ML_VERSION_PATCH 0

#define UNIFIED_ML_MAKE_VERSION(maj, min, patch) \
  ((maj) * 10000 + (min) * 100 + (patch))

#define UNIFIED_ML_VERSION \
  UNIFIED_ML_MAKE_VERSION(UNIFIED_ML_VERSION_MAJOR, \
           UNIFIED_ML_VERSION_MINOR, \
           UNIFIED_ML_VERSION_PATCH)

#define UNIFIED_ML_VERSION_STRING "1.0.0"
