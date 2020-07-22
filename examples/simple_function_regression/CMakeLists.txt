cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(simple_function_regression)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(simple_function_regression main.cpp)

target_link_libraries(simple_function_regression opennn)

set (source "${CMAKE_CURRENT_SOURCE_DIR}/data")
set (destination "${CMAKE_BINARY_DIR}/examples/data")

file(GLOB MY_PUBLIC_HEADERS
  "${source}/*.*"
)
file(COPY ${MY_PUBLIC_HEADERS} DESTINATION "${destination}")