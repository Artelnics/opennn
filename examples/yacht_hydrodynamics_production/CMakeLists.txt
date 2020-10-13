cmake_minimum_required(VERSION 2.6)

project(opennn_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)


add_executable(yacht_hydrodynamics_production main.cpp)

add_library(libopennn SHARED IMPORTED)
set_property(TARGET libopennn PROPERTY IMPORTED_LOCATION "/usr/local/lib/libopennn.so")

target_link_libraries(yacht_hydrodynamics_production libopennn)