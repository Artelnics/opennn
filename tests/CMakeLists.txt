
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

# The concrete response scenarios drive a real trained network through both solvers, so
# they are far slower than a unit test and report on the results as much as they assert.
# They stay in a separate executable and CTest integration label; full verification
# runs both executables. They can also be selected directly:
#
#     build-ninja/bin/opennn_response_tests --gtest_filter=ConcreteScenario.*

set(RESPONSE_SCENARIO_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/response_optimization/concrete_scenarios_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/response_optimization/feasibility_test.cpp")

# The demo is the same problems written as a user writes them: a plain main, no test
# framework. It has its own main(), so it cannot live in either gtest binary:
#
#     build-ninja/bin/opennn_response_demo

set(RESPONSE_DEMO_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/response_optimization/concrete_demo.cpp")

# The Levenberg Marquardt probe is a study, not a test: it solves two systems of equations
# with Eigen's own solver to see whether that is a shape the feasibility system could be
# written in. It has its own main() as well:
#
#     build-ninja/bin/opennn_lm_probe

set(RESPONSE_PROBE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/response_optimization/levenberg_marquardt_probe.cpp")

# toy_nonlinear.cpp is one concrete scenario cut down to two variables and a formula in
# place of the network. It needs no trained network and runs in milliseconds, so unlike the
# scenarios it stays in opennn_tests:
#
#     build-ninja/bin/opennn_tests --gtest_filter=ToyNonlinearSystem.*

list(REMOVE_ITEM TEST_SOURCES
    ${RESPONSE_SCENARIO_SOURCES} ${RESPONSE_DEMO_SOURCES} ${RESPONSE_PROBE_SOURCES})

add_executable(opennn_tests ${TEST_SOURCES})

target_link_libraries(opennn_tests PRIVATE GTest::gtest opennn)

if(WIN32)
    target_link_libraries(opennn_tests PRIVATE ws2_32)
endif()

add_executable(opennn_response_tests ${RESPONSE_SCENARIO_SOURCES})

target_link_libraries(opennn_response_tests PRIVATE GTest::gtest_main opennn)

target_precompile_headers(opennn_response_tests PRIVATE pch.h)

# The scenarios read the pre-trained network from the source tree, next to the example
# that shares it. Loading the .json needs its .bin sibling, so both are given at once.

target_compile_definitions(opennn_response_tests PRIVATE
    CONCRETE_NETWORK_DIR="${PROJECT_SOURCE_DIR}/examples/concrete")

add_executable(opennn_response_demo ${RESPONSE_DEMO_SOURCES})

target_link_libraries(opennn_response_demo opennn)

target_compile_definitions(opennn_response_demo PRIVATE
    CONCRETE_NETWORK_DIR="${PROJECT_SOURCE_DIR}/examples/concrete")

add_executable(opennn_lm_probe ${RESPONSE_PROBE_SOURCES})

target_link_libraries(opennn_lm_probe opennn)

target_compile_definitions(opennn_lm_probe PRIVATE
    CONCRETE_NETWORK_DIR="${PROJECT_SOURCE_DIR}/examples/concrete")

# OpenNN_ONEDNN_FOUND, not OpenNN_ENABLE_ONEDNN: the latter is a tri-state
# whose default is the string AUTO, and if("AUTO") is true in CMake -- so
# branching on the request would define this on every machine, including the
# ones where oneDNN was looked for and not found.
if(OpenNN_ONEDNN_FOUND)
  target_compile_definitions(opennn_tests PRIVATE OPENNN_TEST_HAS_ONEDNN)
endif()

target_precompile_headers(opennn_tests PRIVATE pch.h)

add_test(NAME run_tests COMMAND opennn_tests)
set(OpenNN_UNIT_TEST_TIMEOUT 600 CACHE STRING "Unit-suite timeout in seconds")
set_tests_properties(run_tests PROPERTIES TIMEOUT ${OpenNN_UNIT_TEST_TIMEOUT})

add_test(NAME response_scenarios COMMAND opennn_response_tests)
set_tests_properties(response_scenarios PROPERTIES TIMEOUT 1800 LABELS integration)
