cmake_minimum_required(VERSION 3.21)
project(baem_v3 VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Compiler flags ──────────────────────────────────────────────────────────
add_compile_options(
    -O3
    -march=native
    -mavx512f
    -mavx512dq
    -ffast-math
    -Wall -Wextra -Wpedantic
)

# ─── OpenMP ──────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ─── unified_ml (external dependency) ───────────────────────────────────────
# Adjust UNIFIED_ML_DIR to the actual path of your unified_ml installation
set(UNIFIED_ML_DIR "" CACHE PATH "Path to unified_ml installation")
if(UNIFIED_ML_DIR)
    find_package(unified_ml REQUIRED PATHS ${UNIFIED_ML_DIR})
    set(HAVE_UNIFIED_ML TRUE)
else()
    message(STATUS "[BAEM] unified_ml not found — building with stub headers")
    set(HAVE_UNIFIED_ML FALSE)
endif()

# ─── Global include dirs ─────────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/poker_core
    ${CMAKE_CURRENT_SOURCE_DIR}/gto_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/baem_tracker
    ${CMAKE_CURRENT_SOURCE_DIR}/baem_learning
    ${CMAKE_CURRENT_SOURCE_DIR}/baem_policy
)

# ─── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(poker_core)
add_subdirectory(gto_engine)
add_subdirectory(baem_tracker)
add_subdirectory(baem_learning)
add_subdirectory(baem_policy)
add_subdirectory(tests)
add_subdirectory(simulation)
add_subdirectory(benchmarks)

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "=== BAEM v3 configuration ===")
message(STATUS "  C++ standard  : C++20")
message(STATUS "  unified_ml    : ${HAVE_UNIFIED_ML}")
message(STATUS "  OpenMP        : ${OpenMP_CXX_VERSION}")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
