#include "pch.h"

#include "../opennn/recurrent_layer.h"

TEST(RecurrentLayerTest, DefaultConstructor)
{
    Recurrent recurrent_layer;

    EXPECT_EQ(recurrent_layer.get_inputs_number(), 0);
    EXPECT_EQ(recurrent_layer.get_outputs_number(), 0);
}


TEST(RecurrentLayerTest, GeneralConstructor)
{
    const Index inputs_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);
    const Index time_steps = get_random_index(1, 10);

    Recurrent recurrent_layer({ inputs_number }, { neurons_number });
    recurrent_layer.set_timesteps(time_steps);

    Index parameters_number = neurons_number + (inputs_number + neurons_number) * neurons_number;
    EXPECT_EQ(recurrent_layer.get_parameters_number(), parameters_number);

    Tensor<type, 1> parameters = recurrent_layer.get_parameters();
    EXPECT_EQ(parameters.size(), parameters_number);

    EXPECT_EQ(recurrent_layer.get_input_dimensions(), dimensions({ time_steps,inputs_number }));
    EXPECT_EQ(recurrent_layer.get_output_dimensions(), dimensions({ neurons_number }));
}


TEST(RecurrentLayerTest, Activations)
{
    Index neurons_number = 4;
    Index samples_number = 3;
    Index inputs_number = 3;
    Index time_steps = 1;

    Recurrent recurrent_layer({ inputs_number }, { neurons_number });
    recurrent_layer.set_parameters_constant(type(1));

    Tensor<type, 2> activations(samples_number, neurons_number);
    Tensor<type, 2> activation_derivatives(samples_number, neurons_number);

    recurrent_layer.set_activation_function(Recurrent::Activation::Logistic);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(0.731), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.196), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::HyperbolicTangent);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(0.761), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.41997), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::Linear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::RectifiedLinear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::ExponentialLinear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::ScaledExponentialLinear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1.05), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1.05), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::SoftPlus);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1.313), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.731), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::SoftSign);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(0.5), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.25), 0.001);

    recurrent_layer.set_activation_function(Recurrent::Activation::HardSigmoid);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(0.7), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.2), 0.001);
}

TEST(RecurrentLayerTest, ForwardPropagate)
{
    // Test HyperbolicTangent

    Index neurons_number = 4;
    Index samples_number = 3;
    Index inputs_number = 3;
    Index time_steps = 5;
    bool is_training = true;

    Recurrent recurrent_layer({ inputs_number }, { neurons_number });

    recurrent_layer.set_activation_function(Recurrent::Activation::HyperbolicTangent);
    recurrent_layer.set_parameters_constant(type(0.1));

    Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
    inputs.setConstant(type(1));

    unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
        = make_unique<RecurrentLayerForwardPropagation>(samples_number, &recurrent_layer);

    pair<type*, dimensions> input_pairs = { inputs.data(), {{samples_number, time_steps, inputs_number}} };

    recurrent_layer.forward_propagate({ input_pairs }, recurrent_layer_forward_propagation, is_training);

    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation_ptr =
        static_cast<RecurrentLayerForwardPropagation*>(recurrent_layer_forward_propagation.get());

    Tensor<type, 2> outputs = recurrent_layer_forward_propagation_ptr->outputs;

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), neurons_number);

    EXPECT_NEAR(outputs(0,0),0.511795,0.01);

    //Test SoftPlus

    samples_number = 3;
    inputs_number = 3;
    neurons_number = 4;

    recurrent_layer.set({inputs_number}, {neurons_number});

    inputs.resize(samples_number, time_steps, inputs_number);
    inputs.setConstant(type(1));

    recurrent_layer.set_activation_function("SoftPlus");
    recurrent_layer.set_timesteps(5);

    recurrent_layer.set_parameters_constant(type(0.1));

    recurrent_layer_forward_propagation = make_unique<RecurrentLayerForwardPropagation>(samples_number, &recurrent_layer);
    input_pairs = { inputs.data(), {{samples_number, time_steps, inputs_number}} };
    recurrent_layer.forward_propagate({ input_pairs }, recurrent_layer_forward_propagation, is_training);

    recurrent_layer_forward_propagation_ptr = static_cast<RecurrentLayerForwardPropagation*>(recurrent_layer_forward_propagation.get());

    outputs = recurrent_layer_forward_propagation_ptr->outputs;

    EXPECT_EQ(outputs.dimensions()[0], samples_number);
    EXPECT_EQ(outputs.dimensions()[1], neurons_number);

    EXPECT_NEAR(outputs(0,0),1.16445,0.01);
}
