//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/models/models.h"

#include <utility>

#include "opennn/core/string_utilities.h"
#include "opennn/neural_network/layers/activation_layer.h"
#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/layers/clamping_layer.h"
#include "opennn/neural_network/layers/c2psa_layer.h"
#include "opennn/neural_network/layers/concatenation_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/grouped_query_attention_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include "opennn/neural_network/layers/non_max_suppression_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/pooling_layer_3d.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/tokenizer_layer.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/neural_network/layers/upsampling_layer.h"

namespace opennn
{

static void finalize_build(NeuralNetwork& network)
{
    network.compile();
    network.set_parameters_glorot();
}

static void add_dense_stack(NeuralNetwork& network,
                            const Shape& complexity_dimensions,
                            const string& hidden_activation)
{
    for (size_t i = 0; i < complexity_dimensions.get_rank(); ++i)
        network.add_layer(make_unique<Dense>(network.get_output_shape(),
                                             Shape{ complexity_dimensions[i] },
                                             hidden_activation,
                                             false,
                                             format("dense_layer_{}", i + 1)));
}

template<typename MakeLayer>
static void add_recurrent_stack(NeuralNetwork& network,
                                const Shape& complexity_dimensions,
                                const string& base_label,
                                MakeLayer make_layer)
{
    const Index layer_count = complexity_dimensions.get_rank();

    for (Index i = 0; i < layer_count; ++i)
    {
        const bool last = (i == layer_count - 1);
        auto layer = make_layer(network.get_output_shape(),
                                Shape{complexity_dimensions[i]},
                                last ? base_label : format("{}_{}", base_label, i + 1));
        if (!last) layer->set_return_sequences(true);
        network.add_layer(std::move(layer));
    }
}

static void add_regression_output(NeuralNetwork& network,
                                  const Shape& output_shape,
                                  const string& output_label,
                                  const char* clamping_method)
{
    network.add_layer(make_unique<Dense>(network.get_output_shape(),
                                         output_shape,
                                         "Identity",
                                         false,
                                         output_label));

    network.add_layer(make_unique<Unscaling>(output_shape));

    auto clamping = make_unique<Clamping>(output_shape);
    if (clamping_method) clamping->set_clamping_method(clamping_method);
    network.add_layer(std::move(clamping));
}

ApproximationNetwork::ApproximationNetwork(const Shape& input_shape,
                                           const Shape& complexity_dimensions,
                                           const Shape& output_shape,
                                           const string& hidden_activation)
    : NeuralNetwork(NetworkTask::Approximation)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, hidden_activation);

    add_regression_output(*this, output_shape, "approximation_layer", nullptr);

    finalize_build(*this);
}

ClassificationNetwork::ClassificationNetwork(const Shape& input_shape,
                                             const Shape& complexity_dimensions,
                                             const Shape& output_shape,
                                             const string& hidden_activation)
    : NeuralNetwork(NetworkTask::Classification)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, hidden_activation);

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    finalize_build(*this);
}

ForecastingNetwork::ForecastingNetwork(const Shape& input_shape,
                                       const Shape& complexity_dimensions,
                                       const Shape& output_shape)
    : NeuralNetwork(NetworkTask::Forecasting)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_recurrent_stack(*this, complexity_dimensions, "recurrent_layer",
                        [](const Shape& in, const Shape& out, const string& label)
                        { return make_unique<Recurrent>(in, out, "Tanh", label); });

    add_regression_output(*this, output_shape, "forecasting_layer", "NoClamping");

    finalize_build(*this);
}

ForecastingLstmNetwork::ForecastingLstmNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape)
    : NeuralNetwork(NetworkTask::Forecasting)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_recurrent_stack(*this, complexity_dimensions, "long_short_term_memory_layer",
                        [](const Shape& in, const Shape& out, const string& label)
                        { return make_unique<LongShortTermMemory>(in, out, "Tanh", "Sigmoid", label); });

    add_regression_output(*this, output_shape, "forecasting_layer", "NoClamping");

    finalize_build(*this);
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape)
    : NeuralNetwork(NetworkTask::AutoAssociation)
{
    throw_if(input_shape.empty(),
             "AutoAssociationNetwork: input shape cannot be empty.");
    throw_if(complexity_dimensions.empty(),
             "AutoAssociationNetwork: complexity dimensions cannot be empty.");

    add_layer(make_unique<Scaling>(input_shape));

    const Shape mapping_shape{ 10 };
    const Shape bottleneck_shape{ complexity_dimensions[0] };

    add_layer(make_unique<Dense>(input_shape,
                                 mapping_shape,
                                 "Tanh",
                                 false,
                                 "mapping_layer"));

    add_layer(make_unique<Dense>(mapping_shape,
                                 bottleneck_shape,
                                 "Identity",
                                 false,
                                 "bottleneck_layer"));

    add_layer(make_unique<Dense>(bottleneck_shape,
                                 mapping_shape,
                                 "Tanh",
                                 false,
                                 "demapping_layer"));

    add_layer(make_unique<Dense>(mapping_shape,
                                 Shape{ output_shape },
                                 "Identity",
                                 false,
                                 "output_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    finalize_build(*this);
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& encoder_dimensions,
                                               const string& hidden_activation,
                                               const string& output_activation)
    : NeuralNetwork(NetworkTask::AutoAssociation)
{
    throw_if(input_shape.empty(),
             "AutoAssociationNetwork: input shape cannot be empty.");
    throw_if(encoder_dimensions.empty(),
             "AutoAssociationNetwork: encoder dimensions cannot be empty.");

    add_layer(make_unique<Scaling>(input_shape));

    for (size_t i = 0; i < encoder_dimensions.get_rank(); ++i)
    {
        throw_if(encoder_dimensions[i] <= 0,
                 "AutoAssociationNetwork: encoder dimensions must be positive.");

        const bool bottleneck = i == encoder_dimensions.get_rank() - 1;
        add_layer(make_unique<Dense>(get_output_shape(),
                                     Shape{encoder_dimensions[i]},
                                     hidden_activation,
                                     false,
                                     bottleneck ? "bottleneck_layer"
                                                : format("encoder_layer_{}", i + 1)));
    }

    Index decoder = 1;
    for (Index i = Index(encoder_dimensions.get_rank()) - 2; i >= 0; --i, ++decoder)
        add_layer(make_unique<Dense>(get_output_shape(),
                                     Shape{encoder_dimensions[i]},
                                     hidden_activation,
                                     false,
                                     format("decoder_layer_{}", decoder)));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 input_shape,
                                 output_activation,
                                 false,
                                 "output_layer"));

    add_layer(make_unique<Unscaling>(input_shape));

    finalize_build(*this);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
