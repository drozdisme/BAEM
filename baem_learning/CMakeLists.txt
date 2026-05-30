add_library(baem_learning INTERFACE)
target_include_directories(baem_learning INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(baem_learning INTERFACE baem_tracker baem_policy)
target_compile_features(baem_learning INTERFACE cxx_std_20)

# unified_ml опциональная зависимость
if(HAVE_UNIFIED_ML)
    target_compile_definitions(baem_learning INTERFACE HAVE_UNIFIED_ML=1)
    target_link_libraries(baem_learning INTERFACE unified_ml::unified_ml)
endif()
