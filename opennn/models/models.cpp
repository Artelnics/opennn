// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/models/models.h"

#include <utility>

#include "opennn/core/string_utilities.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/lstm_layer.h"
#include "opennn/network/layers/recurrent_layer.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/network/layers/unscaling_layer.h"

namespace opennn
{

static void append_dense(Network& network, const Shape& output_shape,
                          const string& activation, const string& label)
{
    network.add_layer(make_unique<Dense>(network.get_output_shape(), output_shape,
                                         activation, BatchNormalization::No, label));
}

static void add_dense_stack(Network& network,
                            const Shape& complexity_dimensions,
                            const string& hidden_activation)
{
    for (size_t i = 0; i < complexity_dimensions.get_rank(); ++i)
        append_dense(network, Shape{complexity_dimensions[i]}, hidden_activation,
                      format("dense_layer_{}", i + 1));
}

static void add_regression_output(Network& network,
                                  const Shape& output_shape,
                                  const string& output_label,
                                  Clamping::ClampingMethod clamping_method
                                      = Clamping::ClampingMethod::Clamping)
{
    append_dense(network, output_shape, "Identity", output_label);

    network.add_layer(make_unique<Unscaling>(output_shape));

    network.add_layer(make_unique<Clamping>(output_shape, clamping_method));
}

template<typename MakeLayer>
static void add_forecasting_layers(Network& network,
                                    const Shape& input_shape,
                                    const Shape& complexity_dimensions,
                                    const Shape& output_shape,
                                    const string& base_label,
                                    MakeLayer make_layer)
{
    network.add_layer(make_unique<Scaling>(input_shape));
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

    add_regression_output(network, output_shape, "forecasting_layer",
                          Clamping::ClampingMethod::NoClamping);
}

ApproximationNetwork::ApproximationNetwork(const Shape& input_shape,
                                           const Shape& complexity_dimensions,
                                           const Shape& output_shape,
                                           const string& hidden_activation)
    : Network(NetworkTask::Approximation)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, hidden_activation);

    add_regression_output(*this, output_shape, "approximation_layer");

    finalize_build();
}

ClassificationNetwork::ClassificationNetwork(const Shape& input_shape,
                                             const Shape& complexity_dimensions,
                                             const Shape& output_shape,
                                             const string& hidden_activation)
    : Network(NetworkTask::Classification)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, hidden_activation);

    append_dense(*this, output_shape, output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                  "classification_layer");

    finalize_build();
}

ForecastingNetwork::ForecastingNetwork(const Shape& input_shape,
                                       const Shape& complexity_dimensions,
                                       const Shape& output_shape)
    : Network(NetworkTask::Forecasting)
{
    add_forecasting_layers(*this, input_shape, complexity_dimensions, output_shape, "recurrent_layer",
                            [](const Shape& in, const Shape& out, const string& label)
                            { return make_unique<Recurrent>(in, out, "Tanh", label); });

    finalize_build();
}

ForecastingLstmNetwork::ForecastingLstmNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape)
    : Network(NetworkTask::Forecasting)
{
    add_forecasting_layers(*this, input_shape, complexity_dimensions, output_shape, "lstm_layer",
                            [](const Shape& in, const Shape& out, const string& label)
                            { return make_unique<LSTM>(in, out, "Tanh", "Sigmoid", label); });

    finalize_build();
}

Autoencoder::Autoencoder(const Shape& input_shape,
                                               const Shape& encoder_dimensions,
                                               const string& hidden_activation,
                                               const string& output_activation)
    : Network(NetworkTask::AnomalyDetection)
{
    throw_if(input_shape.empty(),
             "Autoencoder: input shape cannot be empty.");
    throw_if(encoder_dimensions.empty(),
             "Autoencoder: encoder dimensions cannot be empty.");

    add_layer(make_unique<Scaling>(input_shape));

    for (size_t i = 0; i < encoder_dimensions.get_rank(); ++i)
    {
        throw_if(encoder_dimensions[i] <= 0,
                 "Autoencoder: encoder dimensions must be positive.");

        const bool bottleneck = i == encoder_dimensions.get_rank() - 1;
        append_dense(*this, Shape{encoder_dimensions[i]}, hidden_activation,
                      bottleneck ? "bottleneck_layer" : format("encoder_layer_{}", i + 1));
    }

    Index decoder = 1;
    for (Index i = Index(encoder_dimensions.get_rank()) - 2; i >= 0; --i, ++decoder)
        append_dense(*this, Shape{encoder_dimensions[i]}, hidden_activation,
                      format("decoder_layer_{}", decoder));

    append_dense(*this, input_shape, output_activation, "output_layer");

    add_layer(make_unique<Unscaling>(input_shape));

    finalize_build();
}

}
