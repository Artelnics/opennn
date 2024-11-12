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
/*
    ScalingLayer2D scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_dimensions(), dimensions{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_dimensions(), dimensions{ 1 });
*/
}


/*
namespace opennn
{

ScalingLayer2DTest::ScalingLayer2DTest() : UnitTesting()
{
    scaling_layer.set_display(false);

}


void ScalingLayer2DTest::test_constructor()
{
    cout << "test_constructor\n";

    ScalingLayer2D scaling_layer_1;

    assert_true(scaling_layer_1.get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(scaling_layer_1.get_neurons_number() == 0, LOG);

    ScalingLayer2D scaling_layer_2({ 3 });

    assert_true(scaling_layer_2.get_descriptives().size() == 3, LOG);
    assert_true(scaling_layer_2.get_scaling_methods().size() == 3, LOG);

}


void ScalingLayer2DTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<Descriptives,1> inputs_descriptives;

    pair<type*, dimensions> input_pairs;

    // Test
    
    inputs_number = 1;
    samples_number = 1;
    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<ScalingLayer2D>(dimensions{inputs_number}));

    ForwardPropagation forward_propagation(samples_number, &neural_network);

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer.set({inputs_number});
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::None);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    neural_network.forward_propagate({input_pairs},
                                     forward_propagation);

    outputs = forward_propagation.get_last_trainable_layer_outputs_pair();

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    
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


    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1) - inputs(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(2) - inputs(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    
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

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    
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

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0,0) - type(-1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1,0) - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(2,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    
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

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    type scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation - inputs_descriptives(0).mean / inputs_descriptives(0).standard_deviation;

    assert_true(abs(outputs(0, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

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

    assert_true(outputs.dimension(0) == inputs_number, LOG);
    assert_true(outputs.dimension(1) == samples_number, LOG);

    scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation;

    assert_true(abs(outputs(0, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

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

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation;
    assert_true(abs(outputs(0, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

    scaled_input = inputs(1, 0) / inputs_descriptives(1).standard_deviation;
    assert_true(abs(outputs(1, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

}
*/