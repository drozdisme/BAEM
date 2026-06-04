#pragma once

/**
 * @file unified_ml.hpp
 * @brief Backward-compatible umbrella header for the unified_ml SDK.
 *
 * For production integrations, prefer `#include <unified_ml_stable.hpp>` or
 * `#include <unified_ml>`. This header is retained for compatibility and
 * re-exports both the stable and optional experimental surfaces.
 */

#include "unified_ml_stable.hpp"
#include "unified_ml_experimental.hpp"
