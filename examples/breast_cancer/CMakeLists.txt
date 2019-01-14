# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(airfoil_self_noise)

if (WIN32)
    set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif (WIN32)

if (UNIX)
    set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif (UNIX)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(breast_cancer main.cpp)

target_link_libraries(breast_cancer ${PROJECT_LINK_LIBS})
