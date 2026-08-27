cmake_minimum_required(VERSION 3.24)

project(OpenNN VERSION 8.0.1 LANGUAGES CXX)

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

if(OpenNN_BUILD_TESTS OR OpenNN_CHECK_HEADERS)
    enable_testing()
endif()

add_subdirectory(opennn)

if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(OpenNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(OpenNN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
