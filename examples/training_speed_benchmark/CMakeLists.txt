cmake_minimum_required(VERSION 3.18)

project(training_speed_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_speed
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/training-speed/opennn_speed.cpp)

target_link_libraries(opennn_speed PUBLIC opennn)
