cmake_minimum_required(VERSION 2.8.12)
project(tests)

# Add the googletest subdirectory
add_subdirectory(../googletest ${CMAKE_BINARY_DIR}/googletest)

# Include the Google Test headers
include_directories(../googletest/include)

# Set up the precompiled header
set(PCH_HEADER pch.h)
set(PCH_SOURCE pch.cpp)

# Add the test executable
add_executable(run_tests test.cpp 
                         adaptive_moment_estimation_test.cpp 
                         bounding_layer_test.cpp
                         conjugate_gradient_test.cpp
                         convolutional_layer_test.cpp 
                         correlations_test.cpp
                         cross_entropy_error_3d_test.cpp 
                         cross_entropy_error_test.cpp 
                         data_set_test.cpp 
                         flatten_layer_test.cpp 
                         genetic_algorithm_test.cpp 
                         growing_inputs_test.cpp 
                         growing_neurons_test.cpp 
                         image_data_set_test.cpp 
                         inputs_selection_test.cpp 
                         learning_rate_algorithm_test.cpp 
                         levenberg_marquardt_algorithm_test.cpp 
                         long_short_term_memory_layer_test.cpp 
                         mean_squared_error_test.cpp 
                         minkowski_error_test.cpp 
                         model_selection_test.cpp 
                         neural_network_test.cpp 
                         neurons_selection_test.cpp 
                         normalized_squared_error_test.cpp 
                         ${PCH_SOURCE}                          
                        #  perceptron_layer_test.cpp 
                        #  pooling_layer_test.cpp 
                        #  probabilistic_layer_3d_test.cpp 
                        #  probabilistic_layer_test.cpp 
                        #  quasi_newton_method_test.cpp 
                        #  recurrent_layer_test.cpp 
                        #  response_optimization_test.cpp 
                        #  scaling_layer_test.cpp 
                        #  scaling_test.cpp 
                        #  statistics_test.cpp 
                        #  stochastic_gradient_descent_test.cpp
                        #  sum_squared_error_test.cpp 
                        #  tensor_utilities_test.cpp 
                        #  testing_analysis_test.cpp 
                        #  time_series_data_set_test.cpp 
                        #  training_strategy_test.cpp 
                        #  transformer_test.cpp 
                        #  unscaling_layer_test.cpp 
                        #  weighted_squared_error_test.cpp
                         )

# Link Google Test libraries with your test executable
target_link_libraries(run_tests gtest gtest_main opennn)

# Set precompiled header for the target
target_precompile_headers(run_tests PRIVATE ${PCH_HEADER})
