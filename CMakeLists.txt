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
# MSVC 19.5x (VS 2026) crashes (C1001 in the LTCG "Generating code" pass) when
# optimizing the YOLO gradient loop in loss.cpp under whole-program LTO, so
# default LTO OFF there; it works on GCC/Clang. Override with -DOpenNN_ENABLE_LTO=ON.
if(MSVC)
    set(_opennn_lto_default OFF)
else()
    set(_opennn_lto_default ON)
endif()
option(OpenNN_ENABLE_LTO "Enable interprocedural optimization/LTO for OpenNN release builds" ${_opennn_lto_default})
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

# Silence noisy diagnostics that show up mainly through cudnn-frontend and
# response_optimization switch statements. These are not bugs, just style
# warnings that flooded the GPU build log.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-enum>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-global-constructors)
    add_compile_options(-Wno-exit-time-destructors)
endif()

option(OpenNN_DISABLE_CUDA "Disable CUDA even if toolkit is found" OFF)

# BF16 mixed precision used to be a CMake flag (OPENNN_USE_BF16_ACTIVATIONS).
# It is now a runtime decision: Configuration::set(Device::CUDA, Type::BF16).
# The FP32 and BF16 paths are always compiled in when CUDA is enabled.

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

    # cuDNN frontend (header-only C++ wrapper for the cuDNN graph API).
    # Required for SDPA / Flash Attention. Fetched at configure time; pinned
    # to v1.22.1 (the newest release compatible with cuDNN 9.20).
    include(FetchContent)
    FetchContent_Declare(
        cudnn_frontend
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG        v1.22.1
        GIT_SHALLOW    TRUE
    )
    set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cudnn_frontend)
    set(CUDNN_FRONTEND_INCLUDE_DIR ${cudnn_frontend_SOURCE_DIR}/include)
    message(STATUS "Using cudnn-frontend (FetchContent): ${CUDNN_FRONTEND_INCLUDE_DIR}")
else()
    set(OPENNN_HAS_CUDA OFF CACHE BOOL "")
endif()

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.1
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# Install Eigen's export set alongside OpenNN so the installed OpenNNTargets
# (which links Eigen3::Eigen) resolves via find_dependency(Eigen3). Defaults
# to OFF when Eigen is not the top-level project.
set(EIGEN_BUILD_CMAKE_PACKAGE ON CACHE BOOL "" FORCE)

find_package(Eigen3 CONFIG QUIET)

if(NOT Eigen3_FOUND)
    FetchContent_MakeAvailable(eigen)
endif()

add_subdirectory(opennn)

if(OPENNN_HAS_CUDA)
    option(OpenNN_BUILD_BLANK_CUDA "Build OpenNN CUDA blank example" ON)
endif()

option(OpenNN_BUILD_TESTS "Build OpenNN tests" ON)
if(OpenNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(OpenNN_BUILD_EXAMPLES "Build OpenNN examples" ON)
if(OpenNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmark drivers live next to their write-ups under docs/benchmarks/.
option(OpenNN_BUILD_BENCHMARKS "Build OpenNN benchmark targets" OFF)
if(OpenNN_BUILD_BENCHMARKS)
    add_subdirectory(docs/benchmarks)
endif()
