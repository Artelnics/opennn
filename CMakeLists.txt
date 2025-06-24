cmake_minimum_required(VERSION 3.18)

project(OpenNN_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wno-interference-size)

include(CheckLanguage)
check_language(CUDA)

find_package(OpenMP)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(OPENNN_WITH_CUDA ON CACHE BOOL "")
    find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
    find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include)
    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        set(HAVE_CUDNN ON CACHE BOOL "")
    else()
        set(HAVE_CUDNN OFF CACHE BOOL "")
    endif()
else()
    set(OPENNN_WITH_CUDA OFF CACHE BOOL "")
    set(HAVE_CUDNN OFF CACHE BOOL "")
endif()

add_subdirectory(opennn)

if(OPENNN_WITH_CUDA)
    add_subdirectory(blank_cuda)
endif()

option(OpenNN_BUILD_TESTS "Build OpenNN tests" OFF)
if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
