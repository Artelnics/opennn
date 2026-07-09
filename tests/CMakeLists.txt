cmake_minimum_required(VERSION 3.18)
project(tests)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  GIT_SHALLOW    TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
find_package(GTest 1.11 CONFIG QUIET)

if(NOT GTest_FOUND)
  FetchContent_MakeAvailable(googletest)
endif()

set(PCH_HEADER pch.h)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp")

add_executable(opennn_tests ${TEST_SOURCES})

target_link_libraries(opennn_tests gtest gtest_main opennn)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(opennn_tests stdc++fs)
endif()

target_precompile_headers(opennn_tests PRIVATE ${PCH_HEADER})

add_test(NAME run_tests COMMAND opennn_tests)
set_tests_properties(run_tests PROPERTIES TIMEOUT 600)

