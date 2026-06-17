cmake_minimum_required(VERSION 3.18)

project(resnet50_training_speed_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_resnet50_speed
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/resnet50-training-speed/opennn_resnet50_speed.cpp)

target_link_libraries(opennn_resnet50_speed PUBLIC opennn)
