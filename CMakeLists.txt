cmake_minimum_required(VERSION 3.24)

project(OpenNN VERSION 9.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

option(OpenNN_BUILD_TESTS "Build OpenNN tests" ON)
option(OpenNN_BUILD_EXAMPLES "Build OpenNN examples" ON)
option(OpenNN_BUILD_BENCHMARKS "Build OpenNN benchmark targets" OFF)
option(OpenNN_CHECK_HEADERS "Compile each OpenNN header in isolation" OFF)
option(OpenNN_BUILD_FUZZERS "Build Clang libFuzzer targets" OFF)

if(OpenNN_BUILD_TESTS OR OpenNN_CHECK_HEADERS)
    enable_testing()
endif()

add_subdirectory(opennn)

if(OpenNN_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR MSVC)
        message(FATAL_ERROR "OpenNN_BUILD_FUZZERS requires Clang with libFuzzer support")
    endif()
    add_executable(opennn_json_fuzz tools/fuzz/json_fuzz.cpp)
    target_link_libraries(opennn_json_fuzz PRIVATE opennn)
    target_compile_options(opennn_json_fuzz PRIVATE
        -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(opennn_json_fuzz PRIVATE
        -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
endif()

if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(OpenNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(OpenNN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(OpenNN_INSTALL)
    set(CPACK_PACKAGE_NAME OpenNN)
    set(CPACK_PACKAGE_VENDOR Artelnics)
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenNN C++ neural network library, release candidate")
    set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE.txt")
    set(CPACK_GENERATOR "ZIP;TGZ")
    set(CPACK_PACKAGE_CHECKSUM SHA256)
    get_target_property(_opennn_package_definitions opennn INTERFACE_COMPILE_DEFINITIONS)
    if("OPENNN_HAS_CUDA" IN_LIST _opennn_package_definitions)
        set(_opennn_package_backend cuda)
    else()
        set(_opennn_package_backend cpu)
    endif()
    set(CPACK_PACKAGE_FILE_NAME
        "OpenNN-${PROJECT_VERSION}-candidate-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OpenNN_CPU_TARGET}-${_opennn_package_backend}")
    include(CPack)
endif()
