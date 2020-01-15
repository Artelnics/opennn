# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(rosenbrock)

set (PROJECT_LINK_LIBS ../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(rosenbrock main.cpp)

target_link_libraries(rosenbrock ${PROJECT_LINK_LIBS})


