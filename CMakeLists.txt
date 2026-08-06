cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(OpenNN_Project LANGUAGES CXX)

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/opennn/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    if(WIN32)
        # MinGW/Windows: -march=native enables AVX but GCC emits 32-byte-aligned
        # YMM stack spills while the Windows x64 ABI only guarantees 16-byte stack
        # alignment -> intermittent SIGSEGV (-mstackrealign does not fix it). Use
        # SSE (16-byte packets), which matches the ABI. Kept in sync with nd_flags.
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mno-avx>)
    endif()
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
    # Apple Clang's libc++ gates std::jthread / <stop_token> (used in batch.h)
    # behind the experimental library. Enable it for compile AND link.
    add_compile_options(-fexperimental-library)
    add_link_options(-fexperimental-library)

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
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-interference-size>)
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
    find_package(CUDNN 9.0 REQUIRED)
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

# Only use a system Eigen if it is at least as new as the version we vendor
# (see GIT_TAG above); otherwise fetch it, so we never build against an
# incompatible Eigen that happens to be installed.
find_package(Eigen3 5.0 CONFIG QUIET)

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

# Compiles every opennn/*.h in isolation (CPU mode, -fsyntax-only) to catch
# headers that only build thanks to the force-included pch.h. Opt-in because
# it adds ~100 compiler invocations to the test run.
option(OpenNN_CHECK_HEADERS "Add a ctest that compiles each opennn header in isolation" OFF)
if(OpenNN_CHECK_HEADERS AND NOT MSVC)
    enable_testing()
    add_test(NAME opennn_header_isolation
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/opennn/cmake/check_headers.sh
                ${CMAKE_CXX_COMPILER}
                ${CMAKE_CURRENT_SOURCE_DIR}
                "$<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>")
endif()
