cmake_minimum_required(VERSION 3.18)

project(text_generation)

add_executable(text_generation main.cpp)

add_custom_command(TARGET text_generation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/text_generation"
)

target_link_libraries(text_generation PUBLIC opennn)
