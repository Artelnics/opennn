cmake_minimum_required(VERSION 3.18)

project(image_classification_max_batch_benchmark)

add_executable(opennn_image_classification_maxbatch_trial
    ${CMAKE_SOURCE_DIR}/docs/benchmarks/image-classification-max-batch/opennn_image_classification_maxbatch_trial.cpp)

target_link_libraries(opennn_image_classification_maxbatch_trial PUBLIC opennn)
