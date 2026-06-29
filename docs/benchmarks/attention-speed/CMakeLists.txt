cmake_minimum_required(VERSION 3.18)

project(opennn_attention_speed_benchmarks LANGUAGES CXX)

if(NOT OPENNN_HAS_CUDA)
    message(STATUS "Skipping attention-speed benchmarks: OpenNN was built without CUDA")
    return()
endif()

function(add_opennn_attention_benchmark target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE opennn)
endfunction()

add_opennn_attention_benchmark(
    opennn_transformer_resident
    ${CMAKE_CURRENT_SOURCE_DIR}/opennn_transformer_resident.cpp)

add_opennn_attention_benchmark(
    opennn_transformer_infer
    ${CMAKE_CURRENT_SOURCE_DIR}/opennn_transformer_infer.cpp)

add_opennn_attention_benchmark(
    opennn_attention_validate
    ${CMAKE_CURRENT_SOURCE_DIR}/opennn_attention_validate.cpp)

add_opennn_attention_benchmark(
    opennn_transformer_train
    ${CMAKE_CURRENT_SOURCE_DIR}/opennn_transformer_train.cpp)
