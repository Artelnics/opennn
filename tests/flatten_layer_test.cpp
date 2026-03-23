#include "pch.h"
#include "../opennn/flatten_layer.h"

using namespace opennn;

class FlattenLayerTest : public ::testing::Test
{
protected:
    const Index height = 4;
    const Index width = 4;
    const Index channels = 3;
    const Shape input_shape{ height, width, channels };

    unique_ptr<Flatten<4>> flatten_layer;

    void SetUp() override
    {
        flatten_layer = make_unique<Flatten<4>>(input_shape);
    }
};


TEST_F(FlattenLayerTest, Constructor)
{
    EXPECT_EQ(flatten_layer->get_input_shape(), input_shape);
    EXPECT_EQ(flatten_layer->get_output_shape(), Shape{ height * width * channels });
    EXPECT_EQ(flatten_layer->get_name(), "Flatten4d");
}


TEST_F(FlattenLayerTest, ForwardPropagate)
{
    const Index batch_size = 2;

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<FlattenForwardPropagation<4>>(batch_size, flatten_layer.get());

    forward_propagation->initialize();

    Tensor1 workspace(get_size(forward_propagation->get_workspace_views()));
    link(workspace.data(), forward_propagation->get_workspace_views());

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(1.23f);

    memcpy(forward_propagation->inputs[0].data, inputs_data.data(), inputs_data.size() * sizeof(type));

    flatten_layer->forward_propagate(forward_propagation, false);

    const TensorView output_view = forward_propagation->get_outputs();
    const Shape& output_dims = output_view.shape;

    ASSERT_EQ(output_dims.size(), 2) << "Flatten<4> should produce a 2D tensor (batch, features).";
    EXPECT_EQ(output_dims[0], batch_size);
    EXPECT_EQ(output_dims[1], height * width * channels);

    for(Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.data[i], 1.23f, 1e-6f);
}


TEST_F(FlattenLayerTest, BackPropagate)
{
    const Index batch_size = 2;

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<FlattenForwardPropagation<4>>(batch_size, flatten_layer.get());
    forward_propagation->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(1.0f);

    memcpy(forward_propagation->inputs[0].data, inputs_data.data(), inputs_data.size() * sizeof(type));

    unique_ptr<LayerBackPropagation> back_propagation =
        make_unique<FlattenBackPropagation<4>>(batch_size, flatten_layer.get());
    back_propagation->initialize();

    Tensor1 workspace_bw(get_size(back_propagation->get_workspace_views()));
    link(workspace_bw.data(), back_propagation->get_workspace_views());

    const Index flattened_size = height * width * channels;
    Tensor2 output_derivatives(batch_size, flattened_size);
    output_derivatives.setConstant(1.0f);
    TensorView output_derivatives_view(output_derivatives.data(), Shape{ batch_size, flattened_size });

    flatten_layer->forward_propagate(forward_propagation, true);

    flatten_layer->back_propagate(forward_propagation->inputs, { output_derivatives_view }, forward_propagation, back_propagation);

    const vector<TensorView> input_derivative_views = back_propagation->get_input_gradients();

    ASSERT_EQ(input_derivative_views.size(), 1);
    const TensorView& input_derivative_view = input_derivative_views[0];

    EXPECT_EQ(input_derivative_view.shape[0], batch_size);
    EXPECT_EQ(input_derivative_view.shape[1], height);
    EXPECT_EQ(input_derivative_view.shape[2], width);
    EXPECT_EQ(input_derivative_view.shape[3], channels);

    for (Index i = 0; i < input_derivative_view.size(); ++i)
        EXPECT_NEAR(input_derivative_view.data[i], 1.0f, 1e-7f);
}
