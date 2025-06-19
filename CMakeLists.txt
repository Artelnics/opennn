cmake_minimum_required(VERSION 3.10)

project(OpenNN_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit)

option(OpenNN_BUILD_BLANK "Build OpenNN blank" ON)
option(OpenNN_BUILD_TESTS "Build OpenNN tests" OFF)

add_subdirectory(opennn)

if(OpenNN_BUILD_BLANK)
    add_subdirectory(blank)
endif()

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found. The 'blank_cuda' target will be built.")
    add_subdirectory(blank_cuda)
else()
    message(STATUS "WARNING: CUDA Toolkit not found. Skipping 'blank_cuda' target.")
endif()

if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()