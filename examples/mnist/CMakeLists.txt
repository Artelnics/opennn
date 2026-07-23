cmake_minimum_required(VERSION 3.18)

project(mnist)

add_executable(mnist main.cpp)

add_custom_command(TARGET mnist POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/mnist"
)

target_link_libraries(mnist PRIVATE opennn)