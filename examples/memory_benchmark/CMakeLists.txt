cmake_minimum_required(VERSION 3.18)

project(memory_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_memory
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/memory/opennn_memory.cpp)

target_link_libraries(opennn_memory PUBLIC opennn)

if(WIN32)
    target_link_libraries(opennn_memory PUBLIC psapi)
endif()
