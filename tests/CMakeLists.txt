
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
# They build with everything else, so an API change still breaks the build, but they are
# kept out of opennn_tests and run on demand:
#
#     build-ninja/bin/opennn_response_tests --gtest_filter=ConcreteScenario.*

set(RESPONSE_SCENARIO_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/response_optimization/concrete_scenarios_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/response_optimization/feasibility_test.cpp")

list(REMOVE_ITEM TEST_SOURCES ${RESPONSE_SCENARIO_SOURCES})

add_executable(opennn_tests ${TEST_SOURCES})

target_link_libraries(opennn_tests gtest gtest_main opennn)

add_executable(opennn_response_tests ${RESPONSE_SCENARIO_SOURCES})

target_link_libraries(opennn_response_tests gtest gtest_main opennn)

target_precompile_headers(opennn_response_tests PRIVATE pch.h)

# The scenarios read the pre-trained network from the source tree, next to the example
# that shares it. Loading the .json needs its .bin sibling, so both are given at once.

target_compile_definitions(opennn_response_tests PRIVATE
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
set_tests_properties(run_tests PROPERTIES TIMEOUT 600)
