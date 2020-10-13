# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(tests)


file(GLOB SOURCES "*.cpp")

add_executable(tests ${SOURCES})

add_library(libopennn SHARED IMPORTED)
set_property(TARGET libopennn PROPERTY IMPORTED_LOCATION "/usr/local/lib/libopennn.so")


target_link_libraries(tests libopennn)
