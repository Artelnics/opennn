cmake_minimum_required(VERSION 3.18)

project(kohle_eaf)

add_executable(kohle_eaf main.cpp)

add_custom_command(TARGET kohle_eaf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data"
)

target_link_libraries(kohle_eaf PUBLIC opennn)
