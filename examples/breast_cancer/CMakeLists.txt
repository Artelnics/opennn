# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(breast_cancer)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(breast_cancer main.cpp)

target_link_libraries(breast_cancer ${PROJECT_LINK_LIBS})


