cmake_minimum_required(VERSION 2.8.12)
project(gtests)

# Add the googletest subdirectory
add_subdirectory(googletest)

# Include the Google Test headers
include_directories(googletest/include)

# Add the test executable
add_executable(run_tests test_main.cpp perceptron_layer_test.cpp)  # Add all your test files here

# Link Google Test libraries with your test executable
target_link_libraries(run_tests gtest gtest_main opennn)
