add_library(gto_engine INTERFACE)
target_include_directories(gto_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gto_engine INTERFACE poker_core)
target_compile_features(gto_engine INTERFACE cxx_std_20)
# cfr_engine.hpp is header-only, included via gto_engine interface
