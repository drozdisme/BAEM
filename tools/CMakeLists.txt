# tools/CMakeLists.txt — опциональная сборка через cmake.
# Глобальный CMakeLists форсит -mavx512f; для портативности убираем эти флаги
# у целей tools (на CPU без AVX-512 они ломают сборку).
remove_definitions()
add_compile_options(-Wno-unused-result)

add_executable(train_pokerbench train_pokerbench.cpp)
add_executable(play_pokerbench  play_pokerbench.cpp)

foreach(tgt train_pokerbench play_pokerbench)
    target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR})
    # снять небезопасные для портативности флаги, если они были добавлены глобально
    target_compile_options(${tgt} PRIVATE -O3)
endforeach()
