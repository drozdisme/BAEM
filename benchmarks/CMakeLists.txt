add_executable(bench_hand_evaluator bench_hand_evaluator.cpp)
target_link_libraries(bench_hand_evaluator PRIVATE poker_core)
target_include_directories(bench_hand_evaluator PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(bench_hpc_gates bench_hpc_gates.cpp)
target_link_libraries(bench_hpc_gates
    PRIVATE poker_core gto_engine baem_tracker baem_policy baem_learning simulation OpenMP::OpenMP_CXX)
target_include_directories(bench_hpc_gates PRIVATE ${CMAKE_SOURCE_DIR})
