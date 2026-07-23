//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I S T R Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "activation_layer.h"
#include "dense_layer.h"
#include "scaling_layer.h"
#include "tokenizer_layer.h"
#include "flatten_layer.h"
#include "addition_layer.h"
#include "embedding_layer.h"
#include "normalization_layer_3d.h"
#include "convolutional_layer.h"
#include "detection_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "multihead_attention_layer.h"
#include "grouped_query_attention_layer.h"
#include "recurrent_layer.h"
#include "long_short_term_memory_layer.h"
#include "non_max_suppression_layer.h"
#include "loss.h"
#include "adaptive_moment_estimation.h"
#include "stochastic_gradient_descent.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"

namespace opennn
{

void register_classes()
{
    const Activation activation;
    const Dense dense;
    const Scaling scaling;
    const Tokenizer tokenizer;
    const Addition addition;
#ifndef OPENNN_NO_VISION
    const Flatten flatten;
    const Embedding embedding;
    const MultiHeadAttention multi_head_attention;
    const GroupedQueryAttention grouped_query_attention;
    const Normalization3d normalization_3d;
    const Convolutional convolutional;
    const Detection detection;
    const Pooling pooling;
    const Pooling3d pooling_3d;
#endif
    const Bounding bounding;
    const Unscaling unscaling;
    const Recurrent recurrent;
    const LongShortTermMemory long_short_term_memory;
    const NonMaxSuppression non_max_suppression;

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
