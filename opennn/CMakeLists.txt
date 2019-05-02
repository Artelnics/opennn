# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(opennn)

add_library(opennn
adaptive_moment_estimation.cpp
association_rules.cpp
bounding_layer.cpp
conjugate_gradient.cpp
correlation_analysis.cpp
cross_entropy_error.cpp
data_set.cpp
file_utilities.cpp
genetic_algorithm.cpp
golden_section_order.cpp
gradient_descent.cpp
growing_inputs.cpp
incremental_order.cpp
inputs.cpp
inputs_selection_algorithm.cpp
inputs_trending_layer.cpp
instances.cpp
k_nearest_neighbors.cpp
learning_rate_algorithm.cpp
levenberg_marquardt_algorithm.cpp
loss_index.cpp
mean_squared_error.cpp
minkowski_error.cpp
missing_values.cpp
model_selection.cpp
multilayer_perceptron.cpp
neural_network.cpp
normalized_squared_error.cpp
numerical_differentiation.cpp
numerical_integration.cpp
optimization_algorithm.cpp
order_selection_algorithm.cpp
outputs.cpp
outputs_trending_layer.cpp
perceptron_layer.cpp
principal_components_layer.cpp
probabilistic_layer.cpp
pruning_inputs.cpp
quasi_newton_method.cpp
scaling_layer.cpp
selective_pruning.cpp
simulated_annealing_order.cpp
stochastic_gradient_descent.cpp
sum_squared_error.cpp
testing_analysis.cpp
text_analytics.cpp
tinyxml2.cpp
training_strategy.cpp
unscaling_layer.cpp
variables.cpp
weighted_squared_error.cpp)

