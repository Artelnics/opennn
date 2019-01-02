# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(opennn)

add_library(opennn
variables.cpp 
instances.cpp 
missing_values.cpp 
data_set.cpp 
inputs.cpp 
outputs.cpp 
unscaling_layer.cpp 
scaling_layer.cpp 
inputs_trending_layer.cpp 
outputs_trending_layer.cpp 
probabilistic_layer.cpp 
perceptron_layer.cpp 
neural_network.cpp 
multilayer_perceptron.cpp  
bounding_layer.cpp 
sum_squared_error.cpp 
root_mean_squared_error.cpp 
loss_index.cpp 
normalized_squared_error.cpp 
minkowski_error.cpp 
mean_squared_error.cpp 
weighted_squared_error.cpp 
cross_entropy_error.cpp 
training_strategy.cpp 
training_algorithm.cpp 
training_rate_algorithm.cpp 
random_search.cpp 
quasi_newton_method.cpp 
levenberg_marquardt_algorithm.cpp 
gradient_descent.cpp 
evolutionary_algorithm.cpp 
conjugate_gradient.cpp 
model_selection.cpp 
order_selection_algorithm.cpp 
incremental_order.cpp 
golden_section_order.cpp 
simulated_annealing_order.cpp 
ant_colony_optimization.cpp 
inputs_selection_algorithm.cpp 
growing_inputs.cpp 
pruning_inputs.cpp 
genetic_algorithm.cpp 
testing_analysis.cpp  
numerical_integration.cpp 
numerical_differentiation.cpp 
principal_components_layer.cpp 
threshold_selection_algorithm.cpp 
f1_score_optimization_threshold.cpp 
matthew_correlation_optimization_threshold.cpp 
youden_index_optimization_threshold.cpp 
kappa_coefficient_optimization_threshold.cpp 
roc_curve_optimization_threshold.cpp 
selective_pruning.cpp 
file_utilities.cpp 
association_rules.cpp 
text_analytics.cpp 
tinyxml2.cpp 
correlation_analysis.cpp)

