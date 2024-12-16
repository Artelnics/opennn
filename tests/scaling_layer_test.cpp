#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/forward_propagation.h".h"
#include "../opennn/scaling_layer_2d.h"
#include "../opennn/descriptives.h"
#include "../opennn/statistics.h"


TEST(ScalingLayerTest, DefaultConstructor)
{
    ScalingLayer2D scaling_layer_2d;

    EXPECT_EQ(scaling_layer_2d.get_input_dimensions(), dimensions{0});
    EXPECT_EQ(scaling_layer_2d.get_output_dimensions(), dimensions{0});
}


TEST(ScalingLayerTest, GeneralConstructor)
{
    ScalingLayer2D scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_dimensions(), dimensions{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_dimensions(), dimensions{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_type(), Layer::Type::Scaling2D);
    EXPECT_EQ(scaling_layer_2d.get_descriptives().size(), 1);
    EXPECT_EQ(scaling_layer_2d.get_scaling_methods().size(), 1);
}


TEST(ScalingLayerTest, ForwardPropagate)
{
    const Index inputs_number = 1;
    const Index samples_number = 1;

    ScalingLayer2D scaling_layer_2d({ 1 });
/*
    Tensor<type, 2> outputs;

    Tensor<Descriptives, 1> inputs_descriptives;

    pair<type*, dimensions> input_pairs;

    // Test

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<ScalingLayer2D>(dimensions{ inputs_number }));

    ForwardPropagation forward_propagation(samples_number, &neural_network);

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer_2d.set({ inputs_number });
    scaling_layer_2d.set_display(false);
    scaling_layer_2d.set_scalers(Scaler::None);

    input_pairs = { inputs.data(), {{samples_number, inputs_number}} };
/*
    neural_network.forward_propagate({ input_pairs },
        forward_propagation);

    outputs = forward_propagation.get_last_trainable_layer_outputs_pair();

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0), inputs(0), NUMERIC_LIMITS_MIN);
*/
}

/*
void ScalingLayer2DTest::test_forward_propagate()
{
    
    // Test

    inputs_number = 3 + rand()%10;
    samples_number = 1;

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::None);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer.forward_propagate({input_pairs},
                                    &scaling_layer_forward_propagation, true);

    outputs = scaling_layer_forward_propagation.outputs;


    EXPECT_EQ(outputs.dimension(0) == samples_number);
    EXPECT_EQ(outputs.dimension(1) == inputs_number);

    EXPECT_NEAR(abs(outputs(0) - inputs(0)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(1) - inputs(1)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(2) - inputs(2)) < NUMERIC_LIMITS_MIN);
    
    // Test

    inputs_number = 1;
    samples_number = 1;

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MinimumMaximum);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer.forward_propagate({input_pairs},
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs.dimension(0) == samples_number);
    EXPECT_EQ(outputs.dimension(1) == inputs_number);

    EXPECT_EQ(abs(outputs(0) - inputs(0)) < NUMERIC_LIMITS_MIN);
    
    // Test

    inputs_number = 3;
    samples_number = 3;

    inputs.resize(samples_number, inputs_number);

    inputs.setValues({{type(1),type(1),type(1)},
                    {type(2),type(2),type(2)},
                    {type(3),type(3),type(3)}});

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MinimumMaximum);

    Tensor<Index, 1> all_indices;
    initialize_sequential(all_indices, 0, 1, inputs_number-1);
    
    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer.forward_propagate({input_pairs},
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs.dimension(0) == samples_number);
    EXPECT_EQ(outputs.dimension(1) == inputs_number);

    EXPECT_NEAR(abs(outputs(0,0) - type(-1)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(1,0) - type(0)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(2,0) - type(1)) < NUMERIC_LIMITS_MIN);
    
    // Test

    inputs_number = 2;
    samples_number = 2;

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MeanStandardDeviation);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    initialize_sequential(all_indices, 0, 1, inputs_number-1);

    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer.forward_propagate({input_pairs},
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs.dimension(0) == samples_number);
    EXPECT_EQ(outputs.dimension(1) == inputs_number);

    type scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation - inputs_descriptives(0).mean / inputs_descriptives(0).standard_deviation;

    EXPECT_NEAR(abs(outputs(0, 0) - scaled_input) < NUMERIC_LIMITS_MIN);

    // Test
    
    inputs_number = 2;
    samples_number = 2;

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::StandardDeviation);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({ {type(0),type(0)},
                      {type(2),type(2)}});

    initialize_sequential(all_indices, 0, 1, inputs_number - 1);

    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer.forward_propagate({input_pairs},
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs.dimension(0) == inputs_number);
    EXPECT_EQ(outputs.dimension(1) == samples_number);

    scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation;

    EXPECT_NEAR(abs(outputs(0, 0) - scaled_input) < NUMERIC_LIMITS_MIN);

    // Test

    inputs_number = 2 + rand()%10;
    samples_number = 1;

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::StandardDeviation);

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    initialize_sequential(all_indices, 0, 1, inputs_number - 1);

    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer.forward_propagate({input_pairs},
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs.dimension(0) == samples_number);
    EXPECT_EQ(outputs.dimension(1) == inputs_number);

    scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation;
    EXPECT_NEAR(abs(outputs(0, 0) - scaled_input) < NUMERIC_LIMITS_MIN);

    scaled_input = inputs(1, 0) / inputs_descriptives(1).standard_deviation;
    EXPECT_NEAR(abs(outputs(1, 0) - scaled_input) < NUMERIC_LIMITS_MIN);

}
*/