#pragma once

/**
 * @file system.hpp
 * @brief Internal umbrella header for the UCAO subsystem.
 *
 * This header groups the repository-owned UCAO stack as a coherent internal
 * subsystem. It is intended for internal builds, internal examples, and
 * specialized workflows that need explicit access to the Clifford-algebraic
 * operator system without promoting it into the top-level production SDK
 * umbrella.
 */

#include "ucao.hpp"
#include "engine.hpp"
#include "engine_registry.hpp"
#include "engine_policy.hpp"

namespace ucao::system {
/** @brief Internal UCAO subsystem semantic version as an integer. */
inline constexpr int version = UCAO_VERSION;
/** @brief Internal UCAO subsystem semantic version string. */
inline constexpr const char* version_string = UCAO_VERSION_STRING;
} // namespace ucao::system
