# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.18.0
  GIT_SHALLOW    TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
find_package(GTest 1.11 CONFIG QUIET)

if(NOT GTest_FOUND)
  FetchContent_MakeAvailable(googletest)
endif()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX "-DESKTOP-[^/\\\\]*\\.cpp$")

add_executable(opennn_tests ${TEST_SOURCES})

target_link_libraries(opennn_tests PRIVATE GTest::gtest opennn)

# Keep executable export checks runnable when an unrelated full-suite source
# needs repair. This target includes the same expression tests as opennn_tests.
add_executable(opennn_expression_execution_tests
    network/expression_execution_test.cpp)
target_link_libraries(opennn_expression_execution_tests
    PRIVATE GTest::gtest_main opennn)
target_precompile_headers(opennn_expression_execution_tests PRIVATE common/pch.h)

if(WIN32)
    target_link_libraries(opennn_tests PRIVATE ws2_32)
    target_link_libraries(opennn_expression_execution_tests PRIVATE ws2_32)
endif()

# OpenNN_ONEDNN_FOUND, not OpenNN_ENABLE_ONEDNN: the latter is a tri-state
# whose default is the string AUTO, and if("AUTO") is true in CMake -- so
# branching on the request would define this on every machine, including the
# ones where oneDNN was looked for and not found.
if(OpenNN_ONEDNN_FOUND)
  target_compile_definitions(opennn_tests PRIVATE OPENNN_TEST_HAS_ONEDNN)
endif()

target_precompile_headers(opennn_tests PRIVATE common/pch.h)

add_test(NAME run_tests COMMAND opennn_tests)
set(OpenNN_UNIT_TEST_TIMEOUT 600 CACHE STRING "Unit-suite timeout in seconds")
set_tests_properties(run_tests PROPERTIES TIMEOUT ${OpenNN_UNIT_TEST_TIMEOUT})
