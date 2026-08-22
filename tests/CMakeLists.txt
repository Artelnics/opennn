
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

target_link_libraries(opennn_tests gtest gtest_main opennn)

target_precompile_headers(opennn_tests PRIVATE pch.h)

add_test(NAME run_tests COMMAND opennn_tests)
set_tests_properties(run_tests PROPERTIES TIMEOUT 600)

