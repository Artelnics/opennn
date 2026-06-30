cmake_minimum_required(VERSION 3.18)

project(rosenbrock_max_batch_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_rosenbrock_trial
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/rosenbrock-max-batch/opennn_rosenbrock_trial.cpp)

target_link_libraries(opennn_rosenbrock_trial PUBLIC opennn)

if(WIN32)
    target_link_libraries(opennn_rosenbrock_trial PUBLIC psapi)
endif()
