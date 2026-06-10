cmake_minimum_required(VERSION 3.18)

project(capacity_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_capacity
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/capacity/opennn_capacity.cpp)

target_link_libraries(opennn_capacity PUBLIC opennn)

if(WIN32)
    target_link_libraries(opennn_capacity PUBLIC psapi)
endif()
