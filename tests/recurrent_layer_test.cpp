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

    Recurrent recurrent_layer({ time_steps, inputs_number }, { neurons_number });

    Index parameters_number = neurons_number + (inputs_number + neurons_number) * neurons_number;

    EXPECT_EQ(recurrent_layer.get_parameters_number(), parameters_number);
    EXPECT_EQ(recurrent_layer.get_input_dimensions(), dimensions({ time_steps, inputs_number }));
    EXPECT_EQ(recurrent_layer.get_output_dimensions(), dimensions({ neurons_number }));
}


TEST(RecurrentLayerTest, ForwardPropagate)
{
    Index outputs_number = 4;
    Index samples_number = 3;
    Index inputs_number = 3;
    Index time_steps = 5;
    bool is_training = true;


    // Test HyperbolicTangent


    {
        Recurrent recurrent_layer({ time_steps,inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("HyperbolicTangent");

        vector<ParameterView> parameter_views = recurrent_layer.get_parameter_views();

        TensorMap<Tensor<type, 1>> biases_map(parameter_views[0].data, parameter_views[0].size);
        biases_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> input_weights_map(parameter_views[1].data, inputs_number, outputs_number);
        input_weights_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> recurrent_weights_map(parameter_views[2].data, outputs_number, outputs_number);
        recurrent_weights_map.setConstant(0.1);

        Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
            = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);

        vector<TensorView> input_tensor;
        input_tensor.push_back({ inputs.data(), {{samples_number, time_steps, inputs_number}} });

        recurrent_layer.forward_propagate(input_tensor, recurrent_layer_forward_propagation, is_training);

        TensorView outputs_view = recurrent_layer_forward_propagation.get()->get_output_pair();

        TensorMap<const Tensor<const type, 2>> output_tensor(outputs_view.data,
                                                             outputs_view.dims[0],
                                                             outputs_view.dims[1]
                                                             );


        EXPECT_EQ(output_tensor.dimension(0), samples_number);
        EXPECT_EQ(output_tensor.dimension(1), outputs_number);
        EXPECT_NEAR(output_tensor(0, 0), 0.550479, 1e-6);
    }


    // Test Logistic


    {
        Recurrent recurrent_layer({ time_steps, inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("Logistic");

        vector<ParameterView> parameter_views = recurrent_layer.get_parameter_views();

        TensorMap<Tensor<type, 1>> biases_map(parameter_views[0].data, parameter_views[0].size);
        biases_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> input_weights_map(parameter_views[1].data, inputs_number, outputs_number);
        input_weights_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> recurrent_weights_map(parameter_views[2].data, outputs_number, outputs_number);
        recurrent_weights_map.setConstant(0.1);

        Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
            = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);

        vector<TensorView> input_tensor;
        input_tensor.push_back({ inputs.data(), {{samples_number, time_steps, inputs_number}} });

        recurrent_layer.forward_propagate(input_tensor, recurrent_layer_forward_propagation, is_training);

        TensorView outputs_view = recurrent_layer_forward_propagation.get()->get_output_pair();

        TensorMap<const Tensor<const type, 2>> output_tensor(outputs_view.data,
                                                             outputs_view.dims[0],
                                                             outputs_view.dims[1]
                                                             );

        EXPECT_EQ(output_tensor.dimension(0), samples_number);
        EXPECT_EQ(output_tensor.dimension(1), outputs_number);
        EXPECT_NEAR(output_tensor(0, 0), type(0.66066), type(1e-3));
    }


    //Test Linear


    {
        Recurrent recurrent_layer({ time_steps, inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("Linear");

        vector<ParameterView> parameter_views = recurrent_layer.get_parameter_views();

        TensorMap<Tensor<type, 1>> biases_map(parameter_views[0].data, parameter_views[0].size);
        biases_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> input_weights_map(parameter_views[1].data, inputs_number, outputs_number);
        input_weights_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> recurrent_weights_map(parameter_views[2].data, outputs_number, outputs_number);
        recurrent_weights_map.setConstant(0.1);

        Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
            = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);

        vector<TensorView> input_tensor;
        input_tensor.push_back({ inputs.data(), {{samples_number, time_steps, inputs_number}} });

        recurrent_layer.forward_propagate(input_tensor, recurrent_layer_forward_propagation, is_training);

        TensorView outputs_view = recurrent_layer_forward_propagation.get()->get_output_pair();

        TensorMap<const Tensor<const type, 2>> output_tensor(outputs_view.data,
                                                             outputs_view.dims[0],
                                                             outputs_view.dims[1]
                                                             );

        EXPECT_EQ(output_tensor.dimension(0), samples_number);
        EXPECT_EQ(output_tensor.dimension(1), outputs_number);
        EXPECT_NEAR(output_tensor(0, 0), type(0.65984), type(1e-3));
    }
    

    //Test RectifiedLinear


    {
        Recurrent recurrent_layer({ time_steps, inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("RectifiedLinear");

        vector<ParameterView> parameter_views = recurrent_layer.get_parameter_views();

        TensorMap<Tensor<type, 1>> biases_map(parameter_views[0].data, parameter_views[0].size);
        biases_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> input_weights_map(parameter_views[1].data, inputs_number, outputs_number);
        input_weights_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> recurrent_weights_map(parameter_views[2].data, outputs_number, outputs_number);
        recurrent_weights_map.setConstant(0.1);

        Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
            = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);

        vector<TensorView> input_tensor;
        input_tensor.push_back({ inputs.data(), {{samples_number, time_steps, inputs_number}} });

        recurrent_layer.forward_propagate(input_tensor, recurrent_layer_forward_propagation, is_training);

        TensorView outputs_view = recurrent_layer_forward_propagation.get()->get_output_pair();

        TensorMap<const Tensor<const type, 2>> output_tensor(outputs_view.data,
                                                             outputs_view.dims[0],
                                                             outputs_view.dims[1]
                                                            );


        EXPECT_EQ(output_tensor.dimension(0), samples_number);
        EXPECT_EQ(output_tensor.dimension(1), outputs_number);
        EXPECT_NEAR(output_tensor(0, 0), type(0.65984), type(1e-3));
    }

    

    //Test ScaledExponentialLinear

    {
        Recurrent recurrent_layer({ time_steps,inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("ScaledExponentialLinear");

        vector<ParameterView> parameter_views = recurrent_layer.get_parameter_views();

        TensorMap<Tensor<type, 1>> biases_map(parameter_views[0].data, parameter_views[0].size);
        biases_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> input_weights_map(parameter_views[1].data, inputs_number, outputs_number);
        input_weights_map.setConstant(0.1);

        TensorMap<Tensor<type, 2>> recurrent_weights_map(parameter_views[2].data, outputs_number, outputs_number);
        recurrent_weights_map.setConstant(0.1);

        Tensor<type, 3> inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(1.0);

        unique_ptr<LayerForwardPropagation> recurrent_layer_forward_propagation
            = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);

        vector<TensorView> input_tensor;
        input_tensor.push_back({ inputs.data(), {{samples_number, time_steps, inputs_number}} });

        recurrent_layer.forward_propagate(input_tensor, recurrent_layer_forward_propagation, true);

        TensorView outputs_view = recurrent_layer_forward_propagation.get()->get_output_pair();

        TensorMap<const Tensor<const type, 2>> output_tensor(outputs_view.data,
                                                             outputs_view.dims[0],
                                                             outputs_view.dims[1]
                                                             );

        EXPECT_EQ(output_tensor.dimension(0), samples_number);
        EXPECT_EQ(output_tensor.dimension(1), outputs_number);
        EXPECT_NEAR(output_tensor(0, 0), 0.65984, 1e-5);
    }
}
