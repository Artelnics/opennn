//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   R E G I S T R Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dense_layer.h"
#include "scaling_layer.h"
#include "flatten_layer.h"
#include "addition_layer.h"
#include "embedding_layer.h"
#include "normalization_layer_3d.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "unscaling_layer.h"
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

void register_classes()
{
    const Dense<2> dense_2d;
    const Dense<3> dense_3d;
    const Scaling<2> scaling_2d;
    const Scaling<3> scaling_3d;
    const Scaling<4> scaling_4d;
    const Flatten<2> flatten_2d;
    const Flatten<3> flatten_3d;
    const Flatten<4> flatten_4d;
    const Addition<3> addition_3d;
    const Addition<4> addition_4d;
    const Embedding embedding;
    const MultiHeadAttention multi_head_attention;
    const Normalization3d normalization_3d;
    const Convolutional convolutional;
    const Pooling pooling;
    const Pooling3d pooling_3d;
    const Bounding bounding;
    const Unscaling unscaling;
    const Recurrent recurrent;

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
