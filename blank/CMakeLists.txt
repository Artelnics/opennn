# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(blank)

if (WIN32)
    set (PROJECT_LINK_LIBS ../opennn/Release/opennn)
endif (WIN32)

if (UNIX)
    set (PROJECT_LINK_LIBS ../opennn/Release/opennn)
endif (UNIX)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(blank main.cpp)

target_link_libraries(blank ${PROJECT_LINK_LIBS})
