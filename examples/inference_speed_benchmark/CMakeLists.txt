cmake_minimum_required(VERSION 3.18)

project(inference_speed_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_inference
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/inference-speed/opennn_inference.cpp)

target_link_libraries(opennn_inference PUBLIC opennn)
