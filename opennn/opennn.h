//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N   N E U R A L   N E T W O R K S   L I B R A R Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_H
#define OPENNN_H

#include "config.h"
#include "half.hpp"

// Data set

#include "data_set.h"

// Neural network

#include "config.h"
#include "layer.h"
#include "pooling_layer.h"
#include "convolutional_layer.h"
#include "bounding_layer.h"
#include "perceptron_layer.h"
#include "long_short_term_memory_layer.h"
#include "recurrent_layer.h"
#include "probabilistic_layer.h"
#include "scaling_layer.h"
#include "region_proposal_layer.h"
#include "non_max_suppression_layer.h"
#include "unscaling_layer.h"
#include "flatten_layer.h"
#include "neural_network.h"

// Training strategy

#include "loss_index.h"

#include "cross_entropy_error.h"
#include "mean_squared_error.h"
#include "minkowski_error.h"
#include "normalized_squared_error.h"
#include "sum_squared_error.h"
#include "weighted_squared_error.h"

#include "conjugate_gradient.h"
#include "gradient_descent.h"
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

#include "numerical_differentiation.h"
#include "correlations.h"
#include "response_optimization.h"
#include "opennn_strings.h"
#include "opennn_images.h"
#include "tensor_utilities.h"
#include "statistics.h"
#include "scaling.h"
#include "region_based_object_detector.h"
#include "json_to_xml.h"
#include "text_analytics.h"
#include "codification.h"

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
