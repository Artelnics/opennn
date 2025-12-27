//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A S T E R   H E A D E R   F I L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_H
#define OPENNN_H

// Precompiled header with common includes and Eigen
#include "pch.h"

// Core utilities
#include "tensors.h"
#include "statistics.h"
#include "correlations.h"
#include "scaling.h"
#include "strings_utilities.h"
#include "images.h"
#include "kmeans.h"

// Data handling
#include "dataset.h"
#include "image_dataset.h"
#include "language_dataset.h"
#include "time_series_dataset.h"

// Base layer
#include "layer.h"

// Neural network layers
#include "dense_layer.h"
#include "dense_layer_3d.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "recurrent_layer.h"
#include "embedding_layer.h"
#include "multihead_attention_layer.h"
#include "normalization_layer_3d.h"
#include "addition_layer.h"
#include "flatten_layer.h"

// Scaling layers
#include "scaling_layer_2d.h"
#include "scaling_layer_3d.h"
#include "scaling_layer_4d.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"

// Neural network
#include "neural_network.h"
#include "standard_networks.h"
#include "transformer.h"
#include "vgg16.h"

// Loss functions
#include "loss_index.h"
#include "mean_squared_error.h"
#include "normalized_squared_error.h"
#include "weighted_squared_error.h"
#include "minkowski_error.h"
#include "cross_entropy_error.h"
#include "cross_entropy_error_3d.h"

// Optimization algorithms
#include "optimization_algorithm.h"
#include "stochastic_gradient_descent.h"
#include "adaptive_moment_estimation.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"

// Training
#include "training_strategy.h"

// Model selection
#include "model_selection.h"
#include "inputs_selection.h"
#include "growing_inputs.h"
#include "neurons_selection.h"
#include "growing_neurons.h"
#include "genetic_algorithm.h"

// Testing and analysis
#include "testing_analysis.h"
#include "response_optimization.h"
#include "model_expression.h"

// Registry
#include "registry.h"

#endif // OPENNN_H
