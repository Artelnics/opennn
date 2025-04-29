//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N   N E U R A L   N E T W O R K S   L I B R A R Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_H
#define OPENNN_H

//#include "pch.h"

// Data set

#include "data_set.h"
#include "time_series_data_set.h"
#include "auto_associative_data_set.h"
#include "image_data_set.h"
#include "language_data_set.h"

// Neural network

#include "layer.h"
#include "pooling_layer.h"
#include "convolutional_layer.h"
#include "bounding_layer.h"
#include "perceptron_layer.h"
#include "perceptron_layer_3d.h"
#include "recurrent_layer.h"
#include "probabilistic_layer.h"
#include "probabilistic_layer_3d.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_4d.h"
#include "embedding_layer.h"
#include "multihead_attention_layer.h"
#include "kmeans.h"
#include "unscaling_layer.h"
#include "flatten_layer.h"
#include "neural_network.h"
#include "auto_associative_neural_network.h"
#include "transformer.h"
#include "forward_propagation.h"
#include "flatten_layer_3d.h"
#include "normalization_layer_3d.h"

// Training strategy

#include "loss_index.h"
#include "back_propagation.h"

#include "cross_entropy_error.h"
#include "cross_entropy_error_3d.h"
#include "mean_squared_error.h"
#include "minkowski_error.h"
#include "normalized_squared_error.h"
#include "weighted_squared_error.h"

#include "levenberg_marquardt_algorithm.h"
#include "quasi_newton_method.h"
#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

// Model selection

#include "model_selection.h"
#include "neurons_selection.h"
#include "growing_neurons.h"
#include "inputs_selection.h"
#include "growing_inputs.h"
#include "genetic_algorithm.h"

// Testing analysis

#include "testing_analysis.h"

// Utilities

#include "correlations.h"
#include "response_optimization.h"
#include "images.h"
#include "tensors.h"
#include "statistics.h"
#include "scaling.h"

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
