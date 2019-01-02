# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(logical_operations)

if (WIN32)
    set (PROJECT_LINK_LIBS libopennn.a)
endif (WIN32)

if (UNIX)
    set (PROJECT_LINK_LIBS libopennn.a)
endif (UNIX)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(logical_operations main.cpp)

target_link_libraries(logical_operations ${PROJECT_LINK_LIBS})
