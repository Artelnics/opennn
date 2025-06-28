#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/scaling_layer_2d.h"
#include "../opennn/statistics.h"

using namespace opennn;

TEST(Scaling2dTest, DefaultConstructor)
{
    Scaling2d scaling_layer_2d;

    EXPECT_EQ(scaling_layer_2d.get_input_dimensions(), dimensions{0});
    EXPECT_EQ(scaling_layer_2d.get_output_dimensions(), dimensions{0});
}


TEST(Scaling2dTest, GeneralConstructor)
{
    Scaling2d scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_dimensions(), dimensions{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_dimensions(), dimensions{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_name(), "Scaling2d");
    EXPECT_EQ(scaling_layer_2d.get_descriptives().size(), 1);
    EXPECT_EQ(scaling_layer_2d.get_scaling_methods().size(), 1);
}

TEST(Scaling2dTest, ForwardPropagate)
{
    Index inputs_number = 3;
    Index samples_number = 2;

    Scaling2d scaling_layer_2d({ inputs_number });

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<Descriptives, 1> inputs_descriptives;

    bool is_training = true;

    // Test None

    scaling_layer_2d.set_scalers(Scaler::None);

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(10));

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_2d);

    pair<type*, dimensions> input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    scaling_layer_2d.forward_propagate({ input_pairs }, forward_propagation, is_training);

    pair<type*, dimensions> output_pair = forward_propagation->get_output_pair();

    outputs = tensor_map<2>(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0)), inputs(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(1)), inputs(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(outputs(2)), inputs(2), NUMERIC_LIMITS_MIN);

    // Test MinimumMaximum

    inputs_number = 1;
    samples_number = 3;

    scaling_layer_2d.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    outputs.resize(samples_number,inputs_number);

    inputs.setValues({{type(2)},{type(4)},{type(6)}});

    scaling_layer_2d.set_scalers(Scaler::MinimumMaximum);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                          forward_propagation,
                                          is_training);

    output_pair = forward_propagation->get_output_pair();

    outputs = tensor_map<2>(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0), type(1.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1), type(2.5), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(2), type(3.5), NUMERIC_LIMITS_MIN);

    // Test MeanStandardDeviation

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({ inputs_number });

    vector<Descriptives> custom_descriptives(inputs_number);
    custom_descriptives[0].set(type(-10.0), type(10.0), type(1.0), type(0.5));
    custom_descriptives[1].set(type(-10.0), type(10.0), type(0.5), type(2.0));
    scaling_layer_2d.set_descriptives(custom_descriptives);

    inputs.resize(samples_number, inputs_number);
    outputs.resize(samples_number, inputs_number);

    inputs.setValues({ {type(0),type(0)},
                      {type(2),type(2)} });

    scaling_layer_2d.set_scalers(Scaler::MeanStandardDeviation);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    scaling_layer_2d.forward_propagate({ input_pairs },
                                         forward_propagation,
                                         is_training);

    output_pair = forward_propagation->get_output_pair();

    outputs = tensor_map<2>(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0, 0), type(-2.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(0, 1), type(-0.25), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1, 0), type(2.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1, 1), type(0.75), NUMERIC_LIMITS_MIN);

    // Test StandardDeviation

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({ inputs_number });

    vector<Descriptives> custom_stddev_descriptives(inputs_number);
    custom_stddev_descriptives[0].set(type(-1.0), type(1.0), type(0.0), type(2.0));
    custom_stddev_descriptives[1].set(type(-1.0), type(1.0), type(0.0), type(0.5));
    scaling_layer_2d.set_descriptives(custom_stddev_descriptives);

    inputs.resize(samples_number, inputs_number);
    outputs.resize(samples_number,inputs_number);

    inputs.setValues({ {type(0),type(0)},
                      {type(2),type(2)} });

    scaling_layer_2d.set_scalers(Scaler::StandardDeviation);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    scaling_layer_2d.forward_propagate({ input_pairs },
                                         forward_propagation,
                                         is_training);

    output_pair = forward_propagation->get_output_pair();

    outputs = tensor_map<2>(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(outputs(0, 0), type(0.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(0, 1), type(0.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1, 0), type(1.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(outputs(1, 1), type(4.0), NUMERIC_LIMITS_MIN);

    // Test Logarithm

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({ inputs_number });

    inputs.resize(samples_number, inputs_number);
    outputs.resize(samples_number,inputs_number);

    inputs.setValues({ {type(0),type(0)},
                      {type(2),type(2)} });

    scaling_layer_2d.set_scalers(Scaler::Logarithm);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = { inputs.data(), {{samples_number, inputs_number}} };

    scaling_layer_2d.forward_propagate({ input_pairs },
                                         forward_propagation,
                                         is_training); 

    output_pair = forward_propagation->get_output_pair();

    outputs = tensor_map<2>(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    // outputs(0,0) = log(0 + offset_col0) = log(0 + (abs(0)+1+eps)) approx log(1) = 0
    // outputs(1,0) = log(2 + offset_col0) = log(2 + (abs(0)+1+eps)) approx log(3) = 1.098612
    // outputs(0,1) = log(0 + offset_col1) = log(0 + (abs(0)+1+eps)) approx log(1) = 0
    // outputs(1,1) = log(2 + offset_col1) = log(2 + (abs(0)+1+eps)) approx log(3) = 1.098612

    EXPECT_NEAR(outputs(0, 0), type(0.0), 0.001);
    EXPECT_NEAR(outputs(0, 1), type(0.0), 0.001); 
    EXPECT_NEAR(outputs(1, 0), type(1.098612), 0.001); 
    EXPECT_NEAR(outputs(1, 1), type(1.098612), 0.001);

    // Test ImageMinMax

    inputs_number = 2;
    samples_number = 2;

    scaling_layer_2d.set({inputs_number});

    inputs.resize(samples_number,inputs_number);
    outputs.resize(samples_number,inputs_number);

    inputs.setValues({{type(0),type(255)},
                      {type(100),type(2)}});

    scaling_layer_2d.set_scalers(Scaler::ImageMinMax);

    forward_propagation = make_unique<Scaling2dForwardPropagation>(samples_number, &scaling_layer_2d);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    scaling_layer_2d.forward_propagate({ input_pairs },
                                       forward_propagation,
                                       is_training);

    output_pair = forward_propagation->get_output_pair();

    outputs = tensor_map<2>(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), inputs_number);

    EXPECT_NEAR(abs(outputs(0, 0)),type(0)/255, 0.001);
    EXPECT_NEAR(abs(outputs(0, 1)),type(255)/255, 0.001);
    EXPECT_NEAR(abs(outputs(1, 0)),type(100)/255, 0.001);
    EXPECT_NEAR(abs(outputs(1, 1)),type(2)/255, 0.001);
}
