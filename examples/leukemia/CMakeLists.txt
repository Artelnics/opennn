cmake_minimum_required(VERSION 2.6)

project(opennn_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)


add_executable(leukemia main.cpp)

# libopennn
add_library(libopennn SHARED IMPORTED)
set_property(TARGET libopennn PROPERTY IMPORTED_LOCATION "/usr/local/lib/libopennn.so")

# OpenMP
find_package(OpenMP REQUIRED)


target_link_libraries(leukemia libopennn OpenMP::OpenMP_CXX)