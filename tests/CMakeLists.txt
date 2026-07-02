cmake_minimum_required(VERSION 3.18)
project(tests)

include(FetchContent)


find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
    )

    # Prevent installation of the vendored copy
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

endif()


find_package(Eigen3 CONFIG QUIET)

if(NOT Eigen3_FOUND)

    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        release-5.0.1
    )

    FetchContent_MakeAvailable(eigen)

endif()


set(PCH_HEADER pch.h)
set(PCH_SOURCE pch.cpp)

add_executable(run_tests test.cpp 
		adaptive_moment_estimation_test.cpp 
		bounding_layer_test.cpp
		convolutional_layer_test.cpp 
		correlations_test.cpp
		cross_entropy_error_3d_test.cpp
		cross_entropy_error_test.cpp
		data_set_test.cpp
		dense_layer_test.cpp
		embedding_layer_test.cpp
		flatten_layer_test.cpp
		genetic_algorithm_test.cpp
		growing_inputs_test.cpp
		growing_neurons_test.cpp
		image_data_set_test.cpp
		learning_rate_algorithm_test.cpp
		levenberg_marquardt_algorithm_test.cpp
		mean_squared_error_test.cpp
		minkowski_error_test.cpp
		model_selection_test.cpp
		multihead_attention_layer_test.cpp
		neural_network_test.cpp
		normalization_layer_3d_test.cpp
		normalized_squared_error_test.cpp
		performance_test.cpp
		pooling_layer_test.cpp
		quasi_newton_method_test.cpp
		recurrent_layer_test.cpp
		response_optimization_test.cpp
		scaling_layer_test.cpp
		scaling_test.cpp
		statistics_test.cpp
		stochastic_gradient_descent_test.cpp
		tensors_test.cpp
		testing_analysis_test.cpp
		time_series_data_set_test.cpp
		training_strategy_test.cpp
		transformer_test.cpp
		unscaling_layer_test.cpp
		weighted_squared_error_test.cpp
)

if(OpenNN_SHARED_LIBS)
	target_link_libraries(run_tests PUBLIC opennn)
else()
	target_link_libraries(run_tests PRIVATE opennn)
endif()

target_link_libraries(run_tests PUBLIC Eigen3::Eigen)

target_link_libraries(run_tests PRIVATE GTest::gtest GTest::gtest_main)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(run_tests PUBLIC stdc++fs)
endif()

target_precompile_headers(run_tests PRIVATE ${PCH_HEADER})


