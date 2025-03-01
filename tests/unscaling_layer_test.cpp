#include "pch.h"

#include "../opennn/unscaling_layer.h"
#include "../opennn/scaling_layer_2d.h"
#include "../opennn/descriptives.h"

TEST(UnscalingLayerTest, DefaultConstructor)
{
    UnscalingLayer unscaling_layer;

    EXPECT_EQ(unscaling_layer.get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(unscaling_layer.get_descriptives().size(), 0);
    EXPECT_EQ(unscaling_layer.get_input_dimensions(), dimensions{0});
    EXPECT_EQ(unscaling_layer.get_output_dimensions(), dimensions{0});
}


TEST(UnscalingLayerTest, GeneralConstructor)
{
    UnscalingLayer unscaling_layer({ 3 });

    EXPECT_EQ(unscaling_layer.get_input_dimensions(), dimensions{ 3 });
    EXPECT_EQ(unscaling_layer.get_output_dimensions(), dimensions{ 3 });
    EXPECT_EQ(unscaling_layer.get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(unscaling_layer.get_descriptives().size(), 3);
}

TEST(UnscalingLayerTest, ForwardPropagate)
{
    Index inputs_number = 3;
    Index samples_number = 1;

    UnscalingLayer unscaling_layer({ inputs_number });

    Tensor<type, 2> outputs;

    Tensor<Descriptives, 1> inputs_descriptives;

    //Test None

    unscaling_layer.set_scalers(Scaler::None);

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setConstant(type(10));

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<UnscalingLayerForwardPropagation>(samples_number, &unscaling_layer);

    pair<type*, dimensions> input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    bool is_training = true;

    unscaling_layer.forward_propagate({ input_pairs }, forward_propagation, is_training);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0, 0)), inputs(0, 0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(0, 1)), inputs(0, 1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(0, 2)), inputs(0, 2), NUMERIC_LIMITS_MIN);

    //Test MinimumMaximum

    inputs_number = 1;
    samples_number = 3;

    ScalingLayer2D scaling_layer_2d({ inputs_number });

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(2)},{type(4)},{type(6)}});

    scaling_layer_2d.set_scalers(Scaler::MinimumMaximum);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    unscaling_layer.set({inputs_number});

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(1.5)},{type(2.5)},{type(3.5)}});

    unscaling_layer.set_scalers(Scaler::MinimumMaximum);

    forward_propagation = make_unique<UnscalingLayerForwardPropagation>(samples_number, &unscaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    Tensor<type, 1> minimuns = scaling_layer_2d.get_minimums();
    Tensor<type, 1> maximuns = scaling_layer_2d.get_maximums();
    Tensor<type, 1> means = scaling_layer_2d.get_means();
    Tensor<type, 1> std_devs = scaling_layer_2d.get_standard_deviations();

    Descriptives desc;
    vector<Descriptives> descriptives;

    for(Index i = 0; i < inputs_number; i++){
        type mean=means[i];
        type std_dev=std_devs[i];
        type min_range=minimuns[i];
        type max_range=maximuns[i];
        desc.set(min_range, max_range, mean, std_dev);
        descriptives.push_back(desc);
    }

    unscaling_layer.set_descriptives(descriptives);

    unscaling_layer.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0), type(2), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1), type(4), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(2), type(6), NUMERIC_LIMITS_MIN);

  //Test MeanStandardDeviation

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::MeanStandardDeviation);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    minimuns = scaling_layer_2d.get_minimums();
    maximuns = scaling_layer_2d.get_maximums();
    means = scaling_layer_2d.get_means();
    std_devs = scaling_layer_2d.get_standard_deviations();

    vector<Descriptives> descriptivess;

    for(Index i = 0; i < inputs_number; i++){
        type mean=means[i];
        type std_dev=std_devs[i];
        type min_range=minimuns[i];
        type max_range=maximuns[i];
        desc.set(min_range, max_range, mean, std_dev);
        descriptivess.push_back(desc);
    }

    unscaling_layer.set_descriptives(descriptivess);

    unscaling_layer.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(-0.707), type(-0.707)},{type(0.707), type(0.707)}});

    unscaling_layer.set_scalers(Scaler::MeanStandardDeviation);

    forward_propagation = make_unique<UnscalingLayerForwardPropagation>(samples_number, &unscaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    unscaling_layer.forward_propagate({ input_pairs },
                                      forward_propagation,
                                      is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0,0), type(0), 0.001);
    EXPECT_NEAR(outputs(0,1), type(0), 0.001);
    EXPECT_NEAR(outputs(1,0), type(2), 0.001);
    EXPECT_NEAR(outputs(1,1), type(2), 0.001);

    //Test StandardDeviation

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::StandardDeviation);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    minimuns = scaling_layer_2d.get_minimums();
    maximuns = scaling_layer_2d.get_maximums();
    means = scaling_layer_2d.get_means();
    std_devs = scaling_layer_2d.get_standard_deviations();

    vector<Descriptives> descriptivesss;

    for(Index i = 0; i < inputs_number; i++){
        type mean=means[i];
        type std_dev=std_devs[i];
        type min_range=minimuns[i];
        type max_range=maximuns[i];
        desc.set(min_range, max_range, mean, std_dev);
        descriptivesss.push_back(desc);
    }

    unscaling_layer.set_descriptives(descriptivesss);

    unscaling_layer.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(0), type(0)},{type(1.41421), type(1.41421)}});

    unscaling_layer.set_scalers(Scaler::StandardDeviation);

    forward_propagation = make_unique<UnscalingLayerForwardPropagation>(samples_number, &unscaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    unscaling_layer.forward_propagate({ input_pairs },
                                      forward_propagation,
                                      is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0,0), type(0), 0.001);
    EXPECT_NEAR(outputs(0,1), type(0), 0.001);
    EXPECT_NEAR(outputs(1,0), type(2), 0.001);
    EXPECT_NEAR(outputs(1,1), type(2), 0.001);

    //Test Logarithm

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::Logarithm);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    minimuns = scaling_layer_2d.get_minimums();
    maximuns = scaling_layer_2d.get_maximums();
    means = scaling_layer_2d.get_means();
    std_devs = scaling_layer_2d.get_standard_deviations();

    vector<Descriptives> descriptivessss;

    for(Index i = 0; i < inputs_number; i++){
        type mean=means[i];
        type std_dev=std_devs[i];
        type min_range=minimuns[i];
        type max_range=maximuns[i];
        desc.set(min_range, max_range, mean, std_dev);
        descriptivessss.push_back(desc);
    }

    unscaling_layer.set_descriptives(descriptivessss);

    unscaling_layer.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(0), type(0)},{type(1.0986), type(1.0986)}});

    unscaling_layer.set_scalers(Scaler::Logarithm);

    forward_propagation = make_unique<UnscalingLayerForwardPropagation>(samples_number, &unscaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    unscaling_layer.forward_propagate({ input_pairs },
                                      forward_propagation,
                                      is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0,0), type(1), 0.001);
    EXPECT_NEAR(outputs(0,1), type(1), 0.001);
    EXPECT_NEAR(outputs(1,0), type(3), 0.001);
    EXPECT_NEAR(outputs(1,1), type(3), 0.001);

    //Test ImageMinMax

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::ImageMinMax);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    minimuns = scaling_layer_2d.get_minimums();
    maximuns = scaling_layer_2d.get_maximums();
    means = scaling_layer_2d.get_means();
    std_devs = scaling_layer_2d.get_standard_deviations();

    vector<Descriptives> descriptivesssss;

    for(Index i = 0; i < inputs_number; i++){
        type mean=means[i];
        type std_dev=std_devs[i];
        type min_range=minimuns[i];
        type max_range=maximuns[i];
        desc.set(min_range, max_range, mean, std_dev);
        descriptivesssss.push_back(desc);
    }

    unscaling_layer.set_descriptives(descriptivesssss);

    unscaling_layer.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(0), type(0)},{type(0.0078), type(0.0078)}});

    unscaling_layer.set_scalers(Scaler::ImageMinMax);

    forward_propagation = make_unique<UnscalingLayerForwardPropagation>(samples_number, &unscaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    unscaling_layer.forward_propagate({ input_pairs },
                                      forward_propagation,
                                      is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0,0), type(0), 0.02);
    EXPECT_NEAR(outputs(0,1), type(0), 0.02);
    EXPECT_NEAR(outputs(1,0), type(2), 0.02);
    EXPECT_NEAR(outputs(1,1), type(2), 0.02);

}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
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
