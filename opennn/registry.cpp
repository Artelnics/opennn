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

using LayerFactory = unique_ptr<Layer>(*)();

struct LayerRegistration
{
    LayerType type;
    string_view name;
    LayerFactory factory;
};

template<typename Class>
unique_ptr<Layer> construct_layer()
{
    return make_unique<Class>();
}

#ifndef OPENNN_NO_VISION

unique_ptr<Layer> construct_rms_normalization()
{
    auto layer = make_unique<Normalization3d>();
    layer->set_method(NormalizationMethod::RMS);
    return layer;
}

#define OPENNN_VISION_FACTORY(factory) factory

#else

#define OPENNN_VISION_FACTORY(factory) nullptr

#endif

const vector<LayerRegistration>& layer_registrations()
{
    static const vector<LayerRegistration> registrations = {
        {LayerType::Activation,             "Activation",             construct_layer<Activation>},
        {LayerType::Addition,               "Addition",               construct_layer<Addition>},
        {LayerType::Bounding,               "Bounding",               construct_layer<Bounding>},
        {LayerType::Concatenation,          "Concatenation",          construct_layer<Concatenation>},
        {LayerType::Concatenation,          "Concatenate",            construct_layer<Concatenation>},
        {LayerType::Convolutional,          "Convolutional",
         OPENNN_VISION_FACTORY(construct_layer<Convolutional>)},
        {LayerType::Dense,                  "Dense",                  construct_layer<Dense>},
        {LayerType::Detection,              "Detection",
         OPENNN_VISION_FACTORY(construct_layer<Detection>)},
        {LayerType::DetectionV8,            "DetectionV8",
         OPENNN_VISION_FACTORY(construct_layer<DetectionV8>)},
        {LayerType::Embedding,              "Embedding",
         OPENNN_VISION_FACTORY(construct_layer<Embedding>)},
        {LayerType::Flatten,                "Flatten",
         OPENNN_VISION_FACTORY(construct_layer<Flatten>)},
        {LayerType::LongShortTermMemory,    "LongShortTermMemory",    construct_layer<LongShortTermMemory>},
        {LayerType::MultiHeadAttention,     "MultiHeadAttention",
         OPENNN_VISION_FACTORY(construct_layer<MultiHeadAttention>)},
        {LayerType::Normalization3d,        "Normalization3d",
         OPENNN_VISION_FACTORY(construct_layer<Normalization3d>)},
        {LayerType::RMSNormalization3d,     "RMSNormalization3d",
         OPENNN_VISION_FACTORY(construct_rms_normalization)},
        {LayerType::GroupedQueryAttention,  "GroupedQueryAttention",
         OPENNN_VISION_FACTORY(construct_layer<GroupedQueryAttention>)},
        {LayerType::NonMaxSuppression,      "NonMaxSuppression",      construct_layer<NonMaxSuppression>},
        {LayerType::Pooling,                "Pooling",
         OPENNN_VISION_FACTORY(construct_layer<Pooling>)},
        {LayerType::Pooling3d,              "Pooling3d",
         OPENNN_VISION_FACTORY(construct_layer<Pooling3d>)},
        {LayerType::Recurrent,              "Recurrent",              construct_layer<Recurrent>},
        {LayerType::Scaling,                "Scaling",                construct_layer<Scaling>},
        {LayerType::Tokenizer,              "Tokenizer",              construct_layer<Tokenizer>},
        {LayerType::Unscaling,              "Unscaling",              construct_layer<Unscaling>},
        {LayerType::Upsample,               "Upsample",               construct_layer<Upsample>},
        {LayerType::C2PSA,                  "C2PSA",                  construct_layer<C2PSA>}
    };

    return registrations;
}

#undef OPENNN_VISION_FACTORY

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

const EnumMap<LayerType>& layer_type_map()
{
    static const vector<EnumMap<LayerType>::Entry> entries = []
    {
        vector<EnumMap<LayerType>::Entry> result;
        result.reserve(layer_registrations().size());

        for (const LayerRegistration& registration : layer_registrations())
            result.emplace_back(registration.type, string(registration.name));

        return result;
    }();

    static const EnumMap<LayerType> map{entries};
    return map;
}

const string& layer_type_to_string(LayerType type)
{
    return layer_type_map().to_string(type);
}

LayerType string_to_layer_type(const string& name)
{
    return layer_type_map().from_string(name);
}

const string& Layer::get_name() const
{
    return layer_type_to_string(layer_type);
}

unique_ptr<Layer> create_layer(const string& name)
{
    const vector<LayerRegistration>& registrations = layer_registrations();
    const auto registration = ranges::find(registrations, name,
                                           &LayerRegistration::name);

    if (registration == registrations.end() || !registration->factory)
        throw runtime_error(format("Component not found: {}", name));

    return registration->factory();
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
