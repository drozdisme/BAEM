#pragma once

/**
 * @file unified_ml_experimental.hpp
 * @brief Optional opt-in umbrella header for experimental unified_ml modules.
 *
 * This build currently exposes no experimental modules. The header remains as a
 * stable opt-in extension point so integrators can keep include paths and build
 * logic unchanged as experimental packages appear in future releases.
 */

#include "unified_ml_version.hpp"
#include "unified_ml_stability.hpp"

namespace unified_ml::experimental {
/** @brief Placeholder namespace for future opt-in experimental APIs. */
} // namespace unified_ml::experimental
