cmake_minimum_required(VERSION 3.21)
project(sle_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(sle_engine
    src/engine.cpp
    src/mixer.cpp
    src/fast_ops.cpp
    src/synthesis.cpp
    src/encoding.cpp
    src/full_engine.cpp
    src/topology.cpp
    src/framework.cpp
    src/baked_inference.cpp
    src/topology_search.cpp
    src/perception.cpp
    src/topology_cache.cpp
    src/jit.cpp
    src/jit_batch.cpp
    src/jit_trainer.cpp
    src/validation.cpp
)

target_include_directories(sle_engine PUBLIC include)
target_compile_options(sle_engine PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_include_directories(sle_engine PUBLIC include)


add_executable(sle_example examples/minimal.cpp)
target_link_libraries(sle_example PRIVATE sle_engine)

add_executable(sle_synthesize_xor examples/synthesize_xor.cpp)
target_link_libraries(sle_synthesize_xor PRIVATE sle_engine)

add_executable(sle_synthesize_residual examples/synthesize_residual.cpp)
target_link_libraries(sle_synthesize_residual PRIVATE sle_engine)

add_executable(sle_full_pipeline examples/full_pipeline.cpp)
target_link_libraries(sle_full_pipeline PRIVATE sle_engine)

add_executable(sle_framework_demo examples/framework_demo.cpp)
target_link_libraries(sle_framework_demo PRIVATE sle_engine)

add_executable(test_sle tests/test_sle.cpp)
target_link_libraries(test_sle PRIVATE sle_engine)

enable_testing()
add_test(NAME test_sle COMMAND test_sle)
