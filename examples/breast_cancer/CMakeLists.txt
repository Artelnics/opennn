cmake_minimum_required(VERSION 3.18)

project(breast_cancer)

add_executable(breast_cancer main.cpp)

add_custom_command(TARGET breast_cancer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/breast_cancer"
)

target_link_libraries(breast_cancer PRIVATE opennn)