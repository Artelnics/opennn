//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I S T R Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/registry.h"

#include <format>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "opennn/neural_network/layers/activation_layer.h"
#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/layers/bounding_layer.h"
#include "opennn/neural_network/layers/c2psa_layer.h"
#include "opennn/neural_network/layers/concatenation_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/layers/non_max_suppression_layer.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/tokenizer_layer.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/neural_network/layers/upsample_layer.h"
#ifndef OPENNN_NO_VISION
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/grouped_query_attention_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/pooling_layer_3d.h"
#endif
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/levenberg_marquardt_algorithm.h"
#include "opennn/training_strategy/quasi_newton_method.h"
#include "opennn/training_strategy/stochastic_gradient_descent.h"
#include "opennn/model_selection/genetic_algorithm.h"
#include "opennn/model_selection/growing_inputs.h"

namespace opennn
{

namespace
{

template<typename Base, typename Class>
unique_ptr<Base> construct()
{
    return make_unique<Class>();
}

template<typename Base>
unique_ptr<Base> create(const unordered_map<string_view, unique_ptr<Base>(*)()>& factories,
                        const string& name)
{
    const auto it = factories.find(name);

    if (it == factories.end())
        throw runtime_error(format("Component not found: {}", name));

    return it->second();
}

}

unique_ptr<Layer> create_layer(const string& name)
{
    static const unordered_map<string_view, unique_ptr<Layer>(*)()> factories = {
        {"Activation", construct<Layer, Activation>},
        {"Addition", construct<Layer, Addition>},
        {"Bounding", construct<Layer, Bounding>},
        {"C2PSA", construct<Layer, C2PSA>},
        {"Concatenation", construct<Layer, Concatenation>},
        {"Concatenate", construct<Layer, Concatenation>},
        {"Dense", construct<Layer, Dense>},
        {"LongShortTermMemory", construct<Layer, LongShortTermMemory>},
        {"NonMaxSuppression", construct<Layer, NonMaxSuppression>},
        {"Recurrent", construct<Layer, Recurrent>},
        {"Scaling", construct<Layer, Scaling>},
        {"Tokenizer", construct<Layer, Tokenizer>},
        {"Unscaling", construct<Layer, Unscaling>},
        {"Upsample", construct<Layer, Upsample>},
#ifndef OPENNN_NO_VISION
        {"Convolutional", construct<Layer, Convolutional>},
        {"Detection", construct<Layer, Detection>},
        {"DetectionV8", construct<Layer, DetectionV8>},
        {"Embedding", construct<Layer, Embedding>},
        {"Flatten", construct<Layer, Flatten>},
        {"GroupedQueryAttention", construct<Layer, GroupedQueryAttention>},
        {"MultiHeadAttention", construct<Layer, MultiHeadAttention>},
        {"Normalization3d", construct<Layer, Normalization3d>},
        {"RMSNormalization3d", []() -> unique_ptr<Layer>
            {
                auto layer = make_unique<Normalization3d>();
                layer->set_method(NormalizationMethod::RMS);
                return layer;
            }},
        {"Pooling", construct<Layer, Pooling>},
        {"Pooling3d", construct<Layer, Pooling3d>},
#endif
    };

    return create(factories, name);
}

unique_ptr<Optimizer> create_optimizer(const string& name)
{
    static const unordered_map<string_view, unique_ptr<Optimizer>(*)()> factories = {
        {"AdaptiveMomentEstimation", construct<Optimizer, AdaptiveMomentEstimation>},
        {"LevenbergMarquardt", construct<Optimizer, LevenbergMarquardtAlgorithm>},
        {"QuasiNewtonMethod", construct<Optimizer, QuasiNewtonMethod>},
        {"StochasticGradientDescent", construct<Optimizer, StochasticGradientDescent>},
    };

    return create(factories, name);
}

unique_ptr<InputsSelection> create_inputs_selection(const string& name)
{
    static const unordered_map<string_view, unique_ptr<InputsSelection>(*)()> factories = {
        {"GeneticAlgorithm", construct<InputsSelection, GeneticAlgorithm>},
        {"GrowingInputs", construct<InputsSelection, GrowingInputs>},
    };

    return create(factories, name);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
