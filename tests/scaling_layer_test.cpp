#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/statistics.h"

using namespace opennn;

TEST(ScalingLayerTest, DefaultConstructor)
{
    Scaling<2> scaling_layer_2d({0});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{0});
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{0});
}


TEST(ScalingLayerTest, GeneralConstructor)
{
    Scaling<2> scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_name(), "Scaling2d");
    EXPECT_EQ(scaling_layer_2d.get_descriptives().size(), 1);
}


TEST(ScalingLayerTest, ForwardPropagate)
{
    Index inputs_number = 3;
    Index samples_number = 2;
    bool is_training = true;

    const type TOLERANCE = type(1.0e-4);

    Scaling<2> scaling_layer_2d({ inputs_number });

    // Test None
    {
        scaling_layer_2d.set_scalers("None");
        Tensor2 inputs(samples_number, inputs_number);
        inputs.setConstant(type(10));

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        scaling_layer_2d.forward_propagate(fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 0), 10.0, TOLERANCE);
    }

    // Test MinimumMaximum
    {
        inputs_number = 1;
        samples_number = 3;
        scaling_layer_2d.set({inputs_number});
        scaling_layer_2d.set_scalers("MinimumMaximum");
        scaling_layer_2d.set_min_max_range(0, 1);

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({{type(2)},{type(4)},{type(6)}});

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        scaling_layer_2d.forward_propagate(fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());

        EXPECT_NEAR(outputs(0, 0), type(1.5), TOLERANCE);
        EXPECT_NEAR(outputs(2, 0), type(3.5), TOLERANCE);
    }

    // Test MeanStandardDeviation
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({ inputs_number });
        scaling_layer_2d.set_scalers("MeanStandardDeviation");

        vector<Descriptives> custom_descriptives(inputs_number);
        custom_descriptives[0].set(type(-10.0), type(10.0), type(1.0), type(0.5));
        custom_descriptives[1].set(type(-10.0), type(10.0), type(0.5), type(2.0));
        scaling_layer_2d.set_descriptives(custom_descriptives);

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({ {type(0),type(0)}, {type(2),type(2)} });

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        scaling_layer_2d.forward_propagate(fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());

        EXPECT_NEAR(outputs(0, 0), type(-2.0), TOLERANCE);
        EXPECT_NEAR(outputs(1, 1), type(0.75), TOLERANCE);
    }

    // Test StandardDeviation
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({ inputs_number });
        scaling_layer_2d.set_scalers("StandardDeviation");

        vector<Descriptives> custom_std_des(inputs_number);
        custom_std_des[0].set(type(-1.0), type(1.0), type(0.0), type(2.0));
        custom_std_des[1].set(type(-1.0), type(1.0), type(0.0), type(0.5));
        scaling_layer_2d.set_descriptives(custom_std_des);

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({ {type(0),type(0)}, {type(2),type(2)} });

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        scaling_layer_2d.forward_propagate(fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());

        EXPECT_NEAR(outputs(1, 0), type(1.0), TOLERANCE);
        EXPECT_NEAR(outputs(1, 1), type(4.0), TOLERANCE);
    }

    // Test Logarithm
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({ inputs_number });
        scaling_layer_2d.set_scalers("Logarithm");

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({ {type(1.0), type(1.0)}, {type(exp(1.0)), type(exp(1.0))} });

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        scaling_layer_2d.forward_propagate(fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());

        EXPECT_NEAR(outputs(0, 0), type(0.0), TOLERANCE);
        EXPECT_NEAR(outputs(1, 0), type(1.0), TOLERANCE);
    }

    // Test ImageMinMax
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({inputs_number});
        scaling_layer_2d.set_scalers("ImageMinMax");

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({{type(0),type(255)}, {type(100),type(2)}});

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        memcpy(fw_prop->inputs[0].data, inputs.data(), inputs.size() * sizeof(type));
        scaling_layer_2d.forward_propagate(fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 1), 1.0, TOLERANCE);
        EXPECT_NEAR(outputs(1, 0), 100.0/255.0, TOLERANCE);
    }
}
