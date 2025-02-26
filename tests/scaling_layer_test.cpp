#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/forward_propagation.h"
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
    Index inputs_number = 3;
    Index samples_number = 1;

    ScalingLayer2D scaling_layer_2d({ inputs_number });

    Tensor<type, 2> outputs;

    Tensor<Descriptives, 1> inputs_descriptives;

    //Test None
    scaling_layer_2d.set_scalers(Scaler::None);

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setConstant(type(10));

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    pair<type*, dimensions> input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    bool is_training = true;

    scaling_layer_2d.forward_propagate({ input_pairs }, forward_propagation, is_training);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0)), inputs(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(1)), inputs(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(2)), inputs(2), NUMERIC_LIMITS_MIN);

    //Test MinimumMaximum

    inputs_number = 1;
    samples_number = 3;

    scaling_layer_2d.set({inputs_number});

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(2)},{type(4)},{type(6)}});

    scaling_layer_2d.set_scalers(Scaler::MinimumMaximum);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.set_parameters_constant(type(1));

    scaling_layer_2d.forward_propagate({ input_pairs },
                                          forward_propagation,
                                          is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1), type(2.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(2), type(3.5), NUMERIC_LIMITS_MIN);

    //Test MeanStandardDeviation

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::MeanStandardDeviation);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0, 0)),type(0.707), 0.001);

    //Test StandardDeviation

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs(samples_number,inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::StandardDeviation);

    forward_propagation = make_unique<ScalingLayer2DForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0, 0)),type(0), 0.001);
    EXPECT_NEAR(abs(outputs(0, 1)),type(1.41421), 0.001);
    EXPECT_NEAR(abs(outputs(1, 0)),type(1.41421), 0.001);
    EXPECT_NEAR(abs(outputs(1, 1)),type(0), 0.001);

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

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0, 0)),type(0), 0.001);
    EXPECT_NEAR(abs(outputs(0, 1)),type(1.0986), 0.001);
    EXPECT_NEAR(abs(outputs(1, 0)),type(1.0986), 0.001);
    EXPECT_NEAR(abs(outputs(1, 1)),type(0), 0.001);

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

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0, 0)),type(0), 0.001);
    EXPECT_NEAR(abs(outputs(0, 1)),type(0.0078), 0.001);
    EXPECT_NEAR(abs(outputs(1, 0)),type(0.0078), 0.001);
    EXPECT_NEAR(abs(outputs(1, 1)),type(0), 0.001);

}
