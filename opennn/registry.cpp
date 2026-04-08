//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   R E G I S T R Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "bounding_layer.h"
#include "multihead_attention_layer.h"
#include "recurrent_layer.h"
#include "loss.h"
#include "adaptive_moment_estimation.h"
#include "stochastic_gradient_descent.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"

namespace opennn
{

void reference_all_layers()
{
    reference_dense_layer();
    reference_scaling_layer();
    reference_flatten_layer();
    reference_addition_layer();
}

void register_classes()
{
    const Bounding bounding_layer;
    const MultiHeadAttention multi_head_attention;
    const Recurrent recurrent_layer;
    const Loss loss;
    const AdaptiveMomentEstimation adaptive_moment_estimation;
    const StochasticGradientDescent stochastic_gradient_descent;
    const QuasiNewtonMethod quasi_newton_method;
    const LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
