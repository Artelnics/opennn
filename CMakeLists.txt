
cmake_minimum_required(VERSION 2.8.10)

project(opennn DESCRIPTION "Open Neural Netwok library")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(opennn)
include_directories(opennn)


option(OpenNN_BUILD_TESTS "Build OpenNN tests" OFF)
if(OpenNN_BUILD_TESTS)
   add_subdirectory(tests)
endif(OpenNN_BUILD_TESTS)


# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
