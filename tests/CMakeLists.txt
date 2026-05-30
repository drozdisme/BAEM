add_executable(test_baem_core test_baem_core.cpp)
target_link_libraries(test_baem_core
    PRIVATE poker_core gto_engine baem_tracker baem_policy OpenMP::OpenMP_CXX)
target_include_directories(test_baem_core PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(test_math_validation test_math_validation.cpp)
target_link_libraries(test_math_validation
    PRIVATE poker_core gto_engine baem_tracker baem_policy OpenMP::OpenMP_CXX)
target_include_directories(test_math_validation PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME baem_core COMMAND test_baem_core)
add_test(NAME math_validation COMMAND test_math_validation)
enable_testing()

add_executable(test_week5_6 test_week5_6.cpp)
target_link_libraries(test_week5_6
    PRIVATE poker_core gto_engine baem_tracker baem_policy baem_learning OpenMP::OpenMP_CXX)
target_include_directories(test_week5_6 PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME week5_6 COMMAND test_week5_6)

add_executable(test_week7_8 test_week7_8.cpp)
target_link_libraries(test_week7_8
    PRIVATE poker_core gto_engine baem_tracker baem_policy baem_learning simulation OpenMP::OpenMP_CXX)
target_include_directories(test_week7_8 PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME week7_8 COMMAND test_week7_8)
