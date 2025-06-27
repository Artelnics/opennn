#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/recurrent_layer.h"

using namespace opennn;

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

    Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });
    recurrent_layer.set_timesteps(time_steps);

    Index parameters_number = neurons_number + (inputs_number + neurons_number) * neurons_number;
    EXPECT_EQ(recurrent_layer.get_parameters_number(), parameters_number);

    Tensor<type, 1> parameters;
    recurrent_layer.get_parameters(parameters);
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
/*
    Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });

    Index total_parameters = recurrent_layer.get_parameters_number();

    Tensor<type, 1> new_parameters(total_parameters);
    new_parameters.setConstant(type(1));

    Index index = 0;

    recurrent_layer.set_parameters(new_parameters, index);

    Tensor<type, 2> activations(samples_number, neurons_number);
    Tensor<type, 2> activation_derivatives(samples_number, neurons_number);

    recurrent_layer.set_activation_function(Recurrent::Activation::Logistic);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(0.731), type(1e-3));
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.196), type(1e-3));

    recurrent_layer.set_activation_function(Recurrent::Activation::HyperbolicTangent);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(0.761), type(1e-3));
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.41997), type(1e-3));

    recurrent_layer.set_activation_function(Recurrent::Activation::Linear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1), type(1e-3));
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), type(1e-3));

    recurrent_layer.set_activation_function(Recurrent::Activation::RectifiedLinear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);
    EXPECT_NEAR(activations(0, 0), type(1), type(1e-3));
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), type(1e-3));

    recurrent_layer.set_activation_function(Recurrent::Activation::ExponentialLinear);
    activations.setConstant(type(1));
    recurrent_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), type(1e-3));
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), type(1e-3));
*/
}

TEST(RecurrentLayerTest, ForwardPropagate)
{
    Index neurons_number = 4;
    Index samples_number = 3;
    Index inputs_number = 3;
    Index time_steps = 5;
    bool is_training = true;
/*
    using Activation = Recurrent::Activation;

    Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });

    // Test HyperbolicTangent

    {
        Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });
        recurrent_layer.set_activation_function(Activation::HyperbolicTangent);

        Index total_parameters = recurrent_layer.get_parameters_number();

        Tensor<type, 1> new_parameters(total_parameters);
        new_parameters.setConstant(type(0.1));

        Index index = 0;
        recurrent_layer.set_parameters(new_parameters, index);

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

        EXPECT_NEAR(outputs(0, 0), type(0.500356), NUMERIC_LIMITS_MIN);
    }

    // Test Logistic

    {
        Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });
        recurrent_layer.set_activation_function(Activation::Logistic);

        Index total_parameters = recurrent_layer.get_parameters_number();

        Tensor<type, 1> new_parameters(total_parameters);
        new_parameters.setConstant(type(0.1));

        Index index = 0;
        recurrent_layer.set_parameters(new_parameters, index);

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

        EXPECT_NEAR(outputs(0, 0), type(0.6441), type(1e-3));
    }

    //Test Linear

    {
        Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });
        recurrent_layer.set_activation_function(Activation::Linear);

        Index total_parameters = recurrent_layer.get_parameters_number();

        Tensor<type, 1> new_parameters(total_parameters);
        new_parameters.setConstant(type(0.1));

        Index index = 0;
        recurrent_layer.set_parameters(new_parameters, index);

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

        EXPECT_NEAR(outputs(0, 0), type(0.57004), type(1e-3));
    }

    //Test RectifiedLinear

    {
        Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });
        recurrent_layer.set_activation_function(Activation::RectifiedLinear);

        Index total_parameters = recurrent_layer.get_parameters_number();

        Tensor<type, 1> new_parameters(total_parameters);
        new_parameters.setConstant(type(0.1));

        Index index = 0;
        recurrent_layer.set_parameters(new_parameters, index);

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

        EXPECT_NEAR(outputs(0, 0), type(0.57004), type(1e-3));
    }

    //Test ExponentialLinear

    {
        Recurrent recurrent_layer({ inputs_number, time_steps }, { neurons_number });
        recurrent_layer.set_activation_function(Activation::ExponentialLinear);

        Index total_parameters = recurrent_layer.get_parameters_number();

        Tensor<type, 1> new_parameters(total_parameters);
        new_parameters.setConstant(type(0.1));

        Index index = 0;
        recurrent_layer.set_parameters(new_parameters, index);

        Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
            = make_unique<RecurrentLayerForwardPropagation>(samples_number, &recurrent_layer);

        pair<type*, dimensions> input_pairs = { inputs.data(), {{samples_number, time_steps, inputs_number}} };

        recurrent_layer.forward_propagate({ input_pairs }, recurrent_layer_forward_propagation, is_training);

        RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation_ptr =
            static_cast<RecurrentLayerForwardPropagation*>(recurrent_layer_forward_propagation.get());

    EXPECT_NEAR(outputs(0,0),0.5700,type(1e-3));

        Tensor<type, 2> outputs = recurrent_layer_forward_propagation_ptr->outputs;

        EXPECT_EQ(outputs.dimension(0), samples_number);
        EXPECT_EQ(outputs.dimension(1), neurons_number);

        EXPECT_NEAR(outputs(0, 0), type(0.57004), type(1e-3));
    }
*/
}
