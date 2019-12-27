cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(pima_indians_diabetes)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(pima_indians_diabetes main.cpp)

target_link_libraries(pima_indians_diabetes ${PROJECT_LINK_LIBS})


