cmake_minimum_required(VERSION 3.18)

project(OpenNN_Project LANGUAGES CXX)

include(GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuración específica para macOS
if(APPLE)
    find_path(LIBOMP_INCLUDE_DIR omp.h
        PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    
    find_library(LIBOMP_LIBRARY NAMES omp libomp.dylib
        PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    
    if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
        message(STATUS "Found OpenMP on macOS: ${LIBOMP_LIBRARY}")
        add_compile_options(-Xpreprocessor -fopenmp)
        include_directories(${LIBOMP_INCLUDE_DIR})
        link_libraries(${LIBOMP_LIBRARY})
    else()
        message(WARNING "OpenMP not found. Continuing without OpenMP support.")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-interference-size)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-global-constructors)
    add_compile_options(-Wno-exit-time-destructors)
endif()

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(OPENNN_WITH_CUDA ON CACHE BOOL "")
else()
    set(OPENNN_WITH_CUDA OFF CACHE BOOL "")
endif()

add_subdirectory(opennn)

option(OpenNN_BUILD_BLANK "Build OpenNN blank" ON)
if(OpenNN_BUILD_BLANK)
    add_subdirectory(blank)
endif()

option(OpenNN_BUILD_TESTS "Build OpenNN tests" ON)
if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

option(OpenNN_BUILD_EXAMPLES "Build OpenNN examples" ON)
if(OpenNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()