cmake_minimum_required(VERSION 3.18)

project(transformer_max_batch_benchmark)

# Source lives with the benchmark docs so it ships next to the write-up.
add_executable(opennn_transformer_maxbatch_trial
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/transformer-max-batch/opennn_transformer_maxbatch_trial.cpp)

target_link_libraries(opennn_transformer_maxbatch_trial PUBLIC opennn)
