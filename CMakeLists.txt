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

        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mno-avx>)
    endif()
elseif(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

include(CheckIPOSupported)

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

if(APPLE)

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

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND NOT OpenNN_DISABLE_CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
            "CUDA compute capabilities to target (default: native)")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(OPENNN_HAS_CUDA ON CACHE BOOL "")
    find_package(CUDNN 9.0 REQUIRED)
    set(HAVE_CUDNN ON)

    include(FetchContent)
    FetchContent_Declare(
        cudnn_frontend
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG        v1.27.0
        GIT_SHALLOW    TRUE
    )
    set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cudnn_frontend)
    set(CUDNN_FRONTEND_INCLUDE_DIR ${cudnn_frontend_SOURCE_DIR}/include)
    message(STATUS "Using cudnn-frontend (FetchContent): ${CUDNN_FRONTEND_INCLUDE_DIR}")

    # Off by default: the kernels take minutes each to compile and only exist
    # for Ampere through Hopper. See cmake/flash_attention.cmake.
    option(OpenNN_WITH_FLASH_ATTENTION
           "Build FlashAttention-2 kernels and let attention take them" OFF)
    set(OpenNN_FLASH_ATTENTION_HEAD_DIMS "32;64;128" CACHE STRING
        "Head dimensions to build FlashAttention-2 kernels for")
    set(OpenNN_FLASH_ATTENTION_SOURCE_DIR "" CACHE PATH
        "An existing flash-attention checkout (empty: one is fetched)")

    if(OpenNN_WITH_FLASH_ATTENTION)
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/flash_attention.cmake)
        opennn_add_flash_attention_kernels(OPENNN_FLASH_ATTENTION_TARGET)
    endif()
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

set(EIGEN_BUILD_CMAKE_PACKAGE ON CACHE BOOL "" FORCE)

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

option(OpenNN_BUILD_BENCHMARKS "Build OpenNN benchmark targets" OFF)
if(OpenNN_BUILD_BENCHMARKS)
    add_subdirectory(docs/benchmarks)
endif()

option(OpenNN_CHECK_HEADERS "Add a ctest that compiles each opennn header in isolation" OFF)
if(OpenNN_CHECK_HEADERS AND NOT MSVC)
    enable_testing()
    add_test(NAME opennn_header_isolation
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/opennn/cmake/check_headers.sh
                ${CMAKE_CXX_COMPILER}
                ${CMAKE_CURRENT_SOURCE_DIR}
                "$<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>")
endif()
