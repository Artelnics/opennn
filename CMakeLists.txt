cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(OpenNN_Project LANGUAGES CXX)

include(GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
elseif(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

include(CheckIPOSupported)
option(OpenNN_ENABLE_LTO "Enable interprocedural optimization/LTO for OpenNN release builds" ON)
if(OpenNN_ENABLE_LTO)
    check_ipo_supported(RESULT OPENNN_LTO_OK)
    if(OPENNN_LTO_OK)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
endif()

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

if(NOT APPLE)
    find_package(OpenMP)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    add_compile_options(-Wno-interference-size)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-global-constructors)
    add_compile_options(-Wno-exit-time-destructors)
endif()

option(OpenNN_DISABLE_CUDA "Disable CUDA even if toolkit is found" OFF)

# BF16 mixed precision used to be a CMake flag (OPENNN_USE_BF16_ACTIVATIONS).
# It is now a runtime decision: Configuration::set(..., TrainingPrecision::BP16,
# InferencePrecision::BP16) — the FP32 and BF16 paths are always compiled in
# (templated kernels) when CUDA is enabled.

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND NOT OpenNN_DISABLE_CUDA)
    # Default to "native" so nvcc compiles directly for the local GPU's
    # compute capability (no JIT step on first kernel launch, tensor-core
    # paths picked at compile time). CMake 3.24+ resolves "native" via the
    # CUDAToolkit at configure time. User can override with -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
            "CUDA compute capabilities to target (default: native)")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(OPENNN_HAS_CUDA ON CACHE BOOL "")

    find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn_version.h cudnn.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        PATHS /usr/local/cuda/include /usr/include)

    find_library(CUDNN_LIBRARY NAMES cudnn
                 HINTS ${CUDAToolkit_LIBRARY_DIR}
                 PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

    if(NOT CUDNN_LIBRARY OR NOT CUDNN_INCLUDE_DIR)
        message(FATAL_ERROR
            "cuDNN not found (library: '${CUDNN_LIBRARY}', headers: '${CUDNN_INCLUDE_DIR}'). "
            "Install cuDNN >= 9.0, or build CPU-only with -DOpenNN_DISABLE_CUDA=ON.")
    endif()

    set(CUDNN_VERSION_HEADER "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    if(NOT EXISTS "${CUDNN_VERSION_HEADER}")
        set(CUDNN_VERSION_HEADER "${CUDNN_INCLUDE_DIR}/cudnn.h")
    endif()

    file(STRINGS "${CUDNN_VERSION_HEADER}" CUDNN_VERSION_LINES
        REGEX "^#define[ \t]+CUDNN_(MAJOR|MINOR|PATCHLEVEL)[ \t]+[0-9]+$")

    foreach(line ${CUDNN_VERSION_LINES})
        if(line MATCHES "CUDNN_MAJOR[ \t]+([0-9]+)")
            set(CUDNN_MAJOR_VERSION "${CMAKE_MATCH_1}")
        elseif(line MATCHES "CUDNN_MINOR[ \t]+([0-9]+)")
            set(CUDNN_MINOR_VERSION "${CMAKE_MATCH_1}")
        elseif(line MATCHES "CUDNN_PATCHLEVEL[ \t]+([0-9]+)")
            set(CUDNN_PATCH_VERSION "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    set(CUDNN_VERSION "${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}")

    if(CUDNN_VERSION VERSION_LESS "9.0")
        message(FATAL_ERROR
            "cuDNN ${CUDNN_VERSION} found at ${CUDNN_LIBRARY}, but >= 9.0 is required "
            "(SDPA / Flash Attention). Upgrade cuDNN, or build CPU-only with "
            "-DOpenNN_DISABLE_CUDA=ON.")
    endif()

    message(STATUS "Found cuDNN ${CUDNN_VERSION}: ${CUDNN_LIBRARY}")
    set(HAVE_CUDNN ON)
    link_libraries(${CUDNN_LIBRARY})

    # cuDNN frontend (header-only C++ wrapper for the cuDNN graph API).
    # Required for SDPA / Flash Attention. Install v1.10+ from
    # https://github.com/NVIDIA/cudnn-frontend or set CUDNN_FRONTEND_INCLUDE_DIR
    # explicitly. Older v1.x releases have API breaks; we target v1.10.x.
    find_path(CUDNN_FRONTEND_INCLUDE_DIR
        NAMES cudnn_frontend.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        PATHS /usr/local/include /usr/include
              ${CMAKE_CURRENT_SOURCE_DIR}/../cudnn-frontend/include
              ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include)

    if(CUDNN_FRONTEND_INCLUDE_DIR)
        message(STATUS "Found cudnn-frontend: ${CUDNN_FRONTEND_INCLUDE_DIR}")
    endif()
else()
    set(OPENNN_HAS_CUDA OFF CACHE BOOL "")
endif()

add_subdirectory(opennn)

option(OpenNN_BUILD_BLANK "Build OpenNN blank" ON)
if(OpenNN_BUILD_BLANK)
    add_subdirectory(blank)
endif()

if(OPENNN_HAS_CUDA)
    add_subdirectory(blank_cuda)
endif()

option(OpenNN_BUILD_TESTS "Build OpenNN tests" OFF)
if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

option(OpenNN_BUILD_EXAMPLES "Build OpenNN examples" ON)
if(OpenNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
