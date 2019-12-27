cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(simple_pattern_recognition)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(simple_pattern_recognition main.cpp)

target_link_libraries(simple_pattern_recognition ${PROJECT_LINK_LIBS})

