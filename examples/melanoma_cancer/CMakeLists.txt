cmake_minimum_required(VERSION 3.18)

project(melanoma_cancer)

add_executable(melanoma_cancer main.cpp)

add_custom_command(TARGET melanoma_cancer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/melanoma_cancer"
)

target_link_libraries(melanoma_cancer PRIVATE opennn)