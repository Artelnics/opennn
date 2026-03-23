#include "pch.h"

#include "../opennn/tensor_utilities.h"
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
    const Index inputs_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);
    const Index time_steps = random_integer(1, 10);

    Recurrent recurrent_layer({ time_steps, inputs_number }, { neurons_number });

    const Index parameters_number = neurons_number + (inputs_number + neurons_number) * neurons_number;

    EXPECT_EQ(recurrent_layer.get_parameters_number(), parameters_number);
    EXPECT_EQ(recurrent_layer.get_input_shape(), Shape({ time_steps, inputs_number }));
    EXPECT_EQ(recurrent_layer.get_output_shape(), Shape({ neurons_number }));
}


TEST(RecurrentLayerTest, ForwardPropagate)
{
    Index outputs_number = 8;
    Index samples_number = 3;
    Index inputs_number = 8;
    Index time_steps = 3;
    bool is_training = true;

    // Test HyperbolicTangent
    {
        Recurrent recurrent_layer({ time_steps, inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("HyperbolicTangent");

        Tensor1 parameters_data(recurrent_layer.get_parameters_number());
        link(parameters_data.data(), recurrent_layer.get_parameter_views());

        vector<TensorView*> parameter_views = recurrent_layer.get_parameter_views();
        TensorMap1(parameter_views[0]->data, parameter_views[0]->size()).setConstant(0.1);
        TensorMap2(parameter_views[1]->data, inputs_number, outputs_number).setConstant(0.1);
        TensorMap2(parameter_views[2]->data, outputs_number, outputs_number).setConstant(0.1);

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);
        fw_prop->initialize();
        Tensor1 workspace_data(get_size(fw_prop->get_workspace_views()));
        link(workspace_data.data(), fw_prop->get_workspace_views());

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        recurrent_layer.forward_propagate(fw_prop, is_training);

        TensorView outputs_view = fw_prop->get_outputs();
        TensorMap2 output_tensor(outputs_view.data, outputs_view.shape[0], outputs_view.shape[1]);

        EXPECT_NEAR(output_tensor(0, 0), 0.924642, 1e-5);
    }

    // Test Sigmoid
    {
        Recurrent recurrent_layer({ time_steps, inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("Sigmoid");

        Tensor1 parameters_data(recurrent_layer.get_parameters_number());
        link(parameters_data.data(), recurrent_layer.get_parameter_views());

        vector<TensorView*> parameter_views = recurrent_layer.get_parameter_views();
        TensorMap1(parameter_views[0]->data, parameter_views[0]->size()).setConstant(0.1);
        TensorMap2(parameter_views[1]->data, inputs_number, outputs_number).setConstant(0.1);
        TensorMap2(parameter_views[2]->data, outputs_number, outputs_number).setConstant(0.1);

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);
        fw_prop->initialize();
        Tensor1 workspace_data(get_size(fw_prop->get_workspace_views()));
        link(workspace_data.data(), fw_prop->get_workspace_views());

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        recurrent_layer.forward_propagate(fw_prop, is_training);

        TensorView outputs_view = fw_prop->get_outputs();
        TensorMap2 output_tensor(outputs_view.data, outputs_view.shape[0], outputs_view.shape[1]);

        EXPECT_NEAR(output_tensor(0, 0), 0.824956, 1e-5);
    }

    // Test Linear
    {
        Recurrent recurrent_layer({ time_steps, inputs_number }, { outputs_number });
        recurrent_layer.set_activation_function("Linear");

        Tensor1 parameters_data(recurrent_layer.get_parameters_number());
        link(parameters_data.data(), recurrent_layer.get_parameter_views());

        vector<TensorView*> parameter_views = recurrent_layer.get_parameter_views();
        TensorMap1(parameter_views[0]->data, parameter_views[0]->size()).setConstant(0.1);
        TensorMap2(parameter_views[1]->data, inputs_number, outputs_number).setConstant(0.1);
        TensorMap2(parameter_views[2]->data, outputs_number, outputs_number).setConstant(0.1);

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<RecurrentForwardPropagation>(samples_number, &recurrent_layer);
        fw_prop->initialize();
        Tensor1 workspace_data(get_size(fw_prop->get_workspace_views()));
        link(workspace_data.data(), fw_prop->get_workspace_views());

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        recurrent_layer.forward_propagate(fw_prop, is_training);

        TensorView outputs_view = fw_prop->get_outputs();
        TensorMap2 output_tensor(outputs_view.data, outputs_view.shape[0], outputs_view.shape[1]);

        EXPECT_NEAR(output_tensor(0, 0), 2.196, 1e-5);
    }
}
