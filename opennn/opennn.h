/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O P E N   N E U R A L   N E T W O R K S   L I B R A R Y                                                    */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __OPENNN_H__
#define __OPENNN_H__

// Data set

#include "data_set.h"
#include "instances.h"
#include "variables.h"
#include "missing_values.h"


// Mathematical model

#include "mathematical_model.h"
#include "ordinary_differential_equations.h"
#include "plug_in.h"


// Model selection

#include "model_selection.h"
#include "order_selection_algorithm.h"
#include "incremental_order.h"
#include "golden_section_order.h"
#include "simulated_annealing_order.h"
#include "inputs_selection_algorithm.h"
#include "selective_pruning.h"
#include "growing_inputs.h"
#include "pruning_inputs.h"
#include "genetic_algorithm.h"
#include "ant_colony_optimization.h"
#include "threshold_selection_algorithm.h"
#include "f1_score_optimization_threshold.h"
#include "matthew_correlation_optimization_threshold.h"
#include "youden_index_optimization_threshold.h"
#include "kappa_coefficient_optimization_threshold.h"
#include "roc_curve_optimization_threshold.h"

// Neural network

#include "conditions_layer.h"
#include "bounding_layer.h"
#include "independent_parameters.h"
#include "inputs.h"
#include "outputs.h"
#include "perceptron_layer.h"
#include "multilayer_perceptron.h"
#include "perceptron.h"
#include "probabilistic_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "neural_network.h"

// Loss index

#include "loss_index.h"
#include "error_term.h"
#include "regularization_term.h"

#include "neural_parameters_norm.h"

#include "cross_entropy_error.h"
#include "mean_squared_error.h"
#include "minkowski_error.h"
#include "normalized_squared_error.h"
#include "root_mean_squared_error.h"
#include "sum_squared_error.h"

// Testing analysis

#include "testing_analysis.h"

// Training strategy

#include "conjugate_gradient.h"
#include "evolutionary_algorithm.h"
#include "gradient_descent.h"
#include "levenberg_marquardt_algorithm.h"
#include "newton_method.h"
#include "quasi_newton_method.h"
#include "random_search.h"
#include "training_algorithm.h"
#include "training_rate_algorithm.h"

// Utilities

#include "matrix.h"
#include "numerical_differentiation.h"
#include "numerical_integration.h"
#include "vector.h"
#include "math.h"

#endif

// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
