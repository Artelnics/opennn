cmake_minimum_required(VERSION 3.18)

project(translation)

add_executable(translation main.cpp)

add_custom_command(TARGET translation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/translation"
)

target_link_libraries(translation PRIVATE opennn)
