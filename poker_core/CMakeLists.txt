add_library(poker_core INTERFACE)
target_include_directories(poker_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(poker_core INTERFACE cxx_std_20)
