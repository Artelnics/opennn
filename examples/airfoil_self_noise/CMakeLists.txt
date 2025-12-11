cmake_minimum_required(VERSION 3.18)

project(airfoil_self_noise)

add_executable(airfoil_self_noise main.cpp)

add_custom_command(TARGET airfoil_self_noise POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data"
)

target_link_libraries(airfoil_self_noise PUBLIC opennn)
