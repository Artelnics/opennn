#include "pch.h"

#include "../opennn/unscaling_layer.h"
#include "../opennn/scaling_layer_2d.h"
#include "../opennn/descriptives.h"
#include "../opennn/statistics.h"

TEST(UnscalingTest, DefaultConstructor)
{
    Unscaling unscaling_layer;

    EXPECT_EQ(unscaling_layer.get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(unscaling_layer.get_descriptives().size(), 0);
    EXPECT_EQ(unscaling_layer.get_input_dimensions(), dimensions{0});
    EXPECT_EQ(unscaling_layer.get_output_dimensions(), dimensions{0});
}


TEST(UnscalingTest, GeneralConstructor)
{
    Unscaling unscaling_layer({ 3 });

    EXPECT_EQ(unscaling_layer.get_input_dimensions(), dimensions{ 3 });
    EXPECT_EQ(unscaling_layer.get_output_dimensions(), dimensions{ 3 });
    EXPECT_EQ(unscaling_layer.get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(unscaling_layer.get_descriptives().size(), 3);
}


TEST(UnscalingTest, ForwardPropagate)
{
    Index inputs_number;
    Index samples_number;
    bool is_training = true;

    Tensor<type, 2> original_data_tensor;
    Tensor<type, 2> scaled_data_tensor;
    Tensor<type, 2> unscaled_data_tensor;

    unique_ptr<LayerForwardPropagation> forward_propagation;
    pair<type*, dimensions> input_pairs;
    pair<type*, dimensions> output_pair;

    // Test None
    inputs_number = 3;
    samples_number = 1;
    Unscaling unscaling_layer_none({ inputs_number });
    unscaling_layer_none.set_scalers(Scaler::None);
    vector<Descriptives> none_descriptives_vec(inputs_number);
    for (Index i = 0; i < inputs_number; ++i)
        none_descriptives_vec[i].set(0, 1, 0.5, 1.0);

    unscaling_layer_none.set_descriptives(none_descriptives_vec);

    original_data_tensor.resize(samples_number, inputs_number);
    original_data_tensor.setConstant(type(10));

    forward_propagation = make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_none);
    input_pairs = { original_data_tensor.data(), {{samples_number, inputs_number}} };
    unscaling_layer_none.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    unscaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_EQ(unscaled_data_tensor.dimension(0), samples_number);
    EXPECT_EQ(unscaled_data_tensor.dimension(1), inputs_number);
    EXPECT_NEAR(unscaled_data_tensor(0, 0), original_data_tensor(0, 0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(0, 1), original_data_tensor(0, 1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(0, 2), original_data_tensor(0, 2), NUMERIC_LIMITS_MIN);

    // Test MinimumMaximum
    inputs_number = 1;
    samples_number = 3;

    Scaling2d scaling_layer_minmax({ inputs_number });
    Unscaling unscaling_layer_minmax({ inputs_number });

    original_data_tensor.resize(samples_number, inputs_number);
    original_data_tensor.setValues({ {type(2)}, {type(4)}, {type(6)} });

    vector<Descriptives> actual_descriptives_minmax = descriptives(original_data_tensor);
    ASSERT_FALSE(actual_descriptives_minmax.empty());
    ASSERT_EQ(actual_descriptives_minmax.size(), inputs_number);
    EXPECT_NEAR(actual_descriptives_minmax[0].minimum, 2.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(actual_descriptives_minmax[0].maximum, 6.0, NUMERIC_LIMITS_MIN);
    EXPECT_GT(abs(actual_descriptives_minmax[0].standard_deviation), NUMERIC_LIMITS_MIN);

    scaling_layer_minmax.set_descriptives(actual_descriptives_minmax);
    scaling_layer_minmax.set_scalers(Scaler::MinimumMaximum);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_minmax);
    input_pairs = { original_data_tensor.data(), {{samples_number, inputs_number}} };
    scaling_layer_minmax.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    scaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(scaled_data_tensor(0, 0), 0.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 0), 0.5, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(2, 0), 1.0, NUMERIC_LIMITS_MIN);

    unscaling_layer_minmax.set_descriptives(actual_descriptives_minmax);
    unscaling_layer_minmax.set_scalers(Scaler::MinimumMaximum);

    forward_propagation = make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_minmax);
    input_pairs = { scaled_data_tensor.data(), {{samples_number, inputs_number}} };
    unscaling_layer_minmax.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    unscaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_EQ(unscaled_data_tensor.dimension(0), samples_number);
    EXPECT_EQ(unscaled_data_tensor.dimension(1), inputs_number);
    EXPECT_NEAR(unscaled_data_tensor(0, 0), 2.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(1, 0), 4.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(2, 0), 6.0, NUMERIC_LIMITS_MIN);

    // Test MeanStandardDeviation
    inputs_number = 2;
    samples_number = 2;

    Scaling2d scaling_layer_msd({ inputs_number });
    Unscaling unscaling_layer_msd({ inputs_number });

    original_data_tensor.resize(samples_number, inputs_number);
    original_data_tensor.setValues({ {type(0), type(10)},
                                    {type(2), type(30)} });

    vector<Descriptives> actual_descriptives_msd = descriptives(original_data_tensor);
    ASSERT_EQ(actual_descriptives_msd.size(), 2);

    type expected_std_dev0_msd = sqrt(2.0);
    type expected_std_dev1_msd = sqrt(200.0);

    EXPECT_NEAR(actual_descriptives_msd[0].mean, 1.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(actual_descriptives_msd[0].standard_deviation, expected_std_dev0_msd, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(actual_descriptives_msd[1].mean, 20.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(actual_descriptives_msd[1].standard_deviation, expected_std_dev1_msd, NUMERIC_LIMITS_MIN);

    scaling_layer_msd.set_descriptives(actual_descriptives_msd);
    scaling_layer_msd.set_scalers(Scaler::MeanStandardDeviation);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_msd);
    input_pairs = { original_data_tensor.data(), {{samples_number, inputs_number}} };
    scaling_layer_msd.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    scaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(scaled_data_tensor(0, 0), (0.0 - actual_descriptives_msd[0].mean) / actual_descriptives_msd[0].standard_deviation, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(0, 1), (10.0 - actual_descriptives_msd[1].mean) / actual_descriptives_msd[1].standard_deviation, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 0), (2.0 - actual_descriptives_msd[0].mean) / actual_descriptives_msd[0].standard_deviation, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 1), (30.0 - actual_descriptives_msd[1].mean) / actual_descriptives_msd[1].standard_deviation, NUMERIC_LIMITS_MIN);

    unscaling_layer_msd.set_descriptives(actual_descriptives_msd);
    unscaling_layer_msd.set_scalers(Scaler::MeanStandardDeviation);

    forward_propagation = make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_msd);
    input_pairs = { scaled_data_tensor.data(), {{samples_number, inputs_number}} };
    unscaling_layer_msd.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    unscaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(unscaled_data_tensor(0, 0), 0.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(0, 1), 10.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(1, 0), 2.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(1, 1), 30.0, NUMERIC_LIMITS_MIN);

    // Test StandardDeviation
    inputs_number = 2;
    samples_number = 2;

    Scaling2d scaling_layer_std({ inputs_number });
    Unscaling unscaling_layer_std({ inputs_number });

    original_data_tensor.resize(samples_number, inputs_number);
    original_data_tensor.setValues({ {type(1),type(10)},
                                    {type(3),type(50)} });

    vector<Descriptives> actual_descriptives_std = descriptives(original_data_tensor);
    ASSERT_EQ(actual_descriptives_std.size(), 2);
    type expected_std_dev0_std = sqrt(2.0);
    type expected_std_dev1_std = sqrt(800.0);

    EXPECT_NEAR(actual_descriptives_std[0].standard_deviation, expected_std_dev0_std, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(actual_descriptives_std[1].standard_deviation, expected_std_dev1_std, NUMERIC_LIMITS_MIN);

    scaling_layer_std.set_descriptives(actual_descriptives_std);
    scaling_layer_std.set_scalers(Scaler::StandardDeviation);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_std);
    input_pairs = { original_data_tensor.data(), {{samples_number, inputs_number}} };
    scaling_layer_std.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    scaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(scaled_data_tensor(0, 0), 1.0 / actual_descriptives_std[0].standard_deviation, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(0, 1), 10.0 / actual_descriptives_std[1].standard_deviation, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 0), 3.0 / actual_descriptives_std[0].standard_deviation, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 1), 50.0 / actual_descriptives_std[1].standard_deviation, NUMERIC_LIMITS_MIN);

    unscaling_layer_std.set_descriptives(actual_descriptives_std);
    unscaling_layer_std.set_scalers(Scaler::StandardDeviation);

    forward_propagation = make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_std);
    input_pairs = { scaled_data_tensor.data(), {{samples_number, inputs_number}} };
    unscaling_layer_std.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    unscaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(unscaled_data_tensor(0, 0), 1.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(0, 1), 10.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(1, 0), 3.0, NUMERIC_LIMITS_MIN);

    EXPECT_NEAR(unscaled_data_tensor(1, 1), 50.0, NUMERIC_LIMITS_MIN * 10);

    // Test Logarithm
    inputs_number = 2;
    samples_number = 2;

    Scaling2d scaling_layer_log({ inputs_number });
    Unscaling unscaling_layer_log({ inputs_number });

    original_data_tensor.resize(samples_number, inputs_number);
    original_data_tensor.setValues({ {type(1), type(std::exp(2.0))},
                                    {type(std::exp(1.0)), type(std::exp(3.0))} });

    vector<Descriptives> dummy_descriptives_log(inputs_number);
    for (Index i = 0; i < inputs_number; ++i)
        dummy_descriptives_log[i].set(0, 1, 0.5, 1.0);

    scaling_layer_log.set_descriptives(dummy_descriptives_log);
    scaling_layer_log.set_scalers(Scaler::Logarithm);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_log);
    input_pairs = { original_data_tensor.data(), {{samples_number, inputs_number}} };
    scaling_layer_log.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    scaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(scaled_data_tensor(0, 0), 0.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(0, 1), 2.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 0), 1.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 1), 3.0, NUMERIC_LIMITS_MIN);

    unscaling_layer_log.set_descriptives(dummy_descriptives_log);
    unscaling_layer_log.set_scalers(Scaler::Logarithm);

    forward_propagation = make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_log);
    input_pairs = { scaled_data_tensor.data(), {{samples_number, inputs_number}} };
    unscaling_layer_log.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    unscaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(unscaled_data_tensor(0, 0), 1.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(0, 1), exp(2.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(1, 0), exp(1.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(unscaled_data_tensor(1, 1), exp(3.0), NUMERIC_LIMITS_MIN);

    // Test ImageMinMax
    inputs_number = 2;
    samples_number = 2;

    Scaling2d scaling_layer_img({ inputs_number });
    Unscaling unscaling_layer_img({ inputs_number });

    original_data_tensor.resize(samples_number, inputs_number);
    original_data_tensor.setValues({ {type(0), type(255)},
                                    {type(127.5), type(51)} });

    vector<Descriptives> dummy_descriptives_img(inputs_number);
    for (Index i = 0; i < inputs_number; ++i)
        dummy_descriptives_img[i].set(0, 255, 127.5, 1.0);

    scaling_layer_img.set_descriptives(dummy_descriptives_img);
    scaling_layer_img.set_scalers(Scaler::ImageMinMax);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_img);
    input_pairs = { original_data_tensor.data(), {{samples_number, inputs_number}} };
    scaling_layer_img.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    scaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(scaled_data_tensor(0, 0), 0.0 / 255.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(0, 1), 255.0 / 255.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 0), 127.5 / 255.0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_tensor(1, 1), 51.0 / 255.0, NUMERIC_LIMITS_MIN);

    unscaling_layer_img.set_descriptives(dummy_descriptives_img);
    unscaling_layer_img.set_scalers(Scaler::ImageMinMax);

    forward_propagation = make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_img);
    input_pairs = { scaled_data_tensor.data(), {{samples_number, inputs_number}} };
    unscaling_layer_img.forward_propagate({ input_pairs }, forward_propagation, is_training);
    output_pair = forward_propagation->get_outputs_pair();
    unscaled_data_tensor = tensor_map_2(output_pair);

    EXPECT_NEAR(unscaled_data_tensor(0, 0), 0.0, NUMERIC_LIMITS_MIN * 40);
    EXPECT_NEAR(unscaled_data_tensor(0, 1), 255.0, NUMERIC_LIMITS_MIN * 40);
    EXPECT_NEAR(unscaled_data_tensor(1, 0), 127.5, NUMERIC_LIMITS_MIN * 40);
    EXPECT_NEAR(unscaled_data_tensor(1, 1), 51.0, NUMERIC_LIMITS_MIN * 40);
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
