# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(airline_passengers)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(airline_passengers main.cpp)

target_link_libraries(airline_passengers ${PROJECT_LINK_LIBS})


