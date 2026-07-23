cmake_minimum_required(VERSION 3.18)

project(iris_plant)

add_executable(iris_plant main.cpp)

add_custom_command(TARGET iris_plant POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/iris_plant"
)

target_link_libraries(iris_plant PRIVATE opennn)