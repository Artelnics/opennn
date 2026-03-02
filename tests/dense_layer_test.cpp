#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"

using namespace opennn;

TEST(Dense2dTest, DefaultConstructor)
{
    opennn::Dense<2> dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense2d");
    EXPECT_EQ(dense_layer.get_input_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_output_shape().size(), 1);
}


TEST(Dense2dTest, GeneralConstructor)
{
    opennn::Dense<2> dense_layer({10}, {3}, "Linear");

    EXPECT_EQ(dense_layer.get_activation_function(), "Linear");

    ASSERT_EQ(dense_layer.get_input_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_input_shape()[0], 10);

    ASSERT_EQ(dense_layer.get_output_shape().size(), 1);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 3);

    EXPECT_EQ(dense_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 4;
    const bool is_training = true;

    opennn::Dense<2> dense2d_layer({ inputs_number }, { outputs_number }, "Linear");
    dense2d_layer.set_parameters_random();

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<DenseForwardPropagation<2>>(batch_size, &dense2d_layer);

    forward_propagation->initialize();

    Tensor2 inputs(batch_size, inputs_number);
    inputs.setConstant(1.0f);

    TensorView input_view(inputs.data(), { batch_size, inputs_number });
    vector<TensorView> input_views = { input_view };

    ASSERT_NO_THROW(
        dense2d_layer.forward_propagate(input_views, forward_propagation, is_training)
        );

    EXPECT_EQ(dense2d_layer.get_name(), "Dense2d");
    EXPECT_EQ(dense2d_layer.get_input_shape()[0], inputs_number);
    EXPECT_EQ(dense2d_layer.get_output_shape()[0], outputs_number);

    const TensorView output_view = forward_propagation->get_outputs();

    ASSERT_EQ(output_view.shape.size(), 2) << "Output should be a 2D tensor.";
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], outputs_number);
}


TEST(Dense3dTest, DefaultConstructor)
{
    opennn::Dense<3> dense_3d;

    EXPECT_EQ(dense_3d.get_name(), "Dense3d");
    EXPECT_TRUE(dense_3d.get_output_shape().size() > 0);
}


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = 5;
    const Index input_embedding = 4;
    const Index output_embedding = 3;

    opennn::Dense<3> dense_3d({sequence_length, input_embedding}, {output_embedding}, "HyperbolicTangent");

    const Shape input_dims = dense_3d.get_input_shape();
    const Shape output_dims = dense_3d.get_output_shape();

    EXPECT_EQ(input_dims[0], sequence_length);
    EXPECT_EQ(output_dims[0], output_embedding);

    EXPECT_EQ(dense_3d.get_name(), "Dense3d");
}


TEST(Dense3dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index input_embedding = 4;
    const Index output_embedding = 5;

    opennn::Dense<3> dense_3d({sequence_length, input_embedding}, {output_embedding});
    dense_3d.set_parameters_random();

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<DenseForwardPropagation<3>>(batch_size, &dense_3d);

    forward_propagation->initialize();

    Tensor3 inputs(batch_size, sequence_length, input_embedding);
    inputs.setConstant(0.5f);

    TensorView input_view(inputs.data(), {batch_size, sequence_length, input_embedding});
    vector<TensorView> input_views = { input_view };

    ASSERT_NO_THROW(
        dense_3d.forward_propagate(input_views, forward_propagation, false)
        );

    const TensorView output_view = forward_propagation->get_outputs();

    ASSERT_EQ(output_view.shape.size(), 3);
    EXPECT_EQ(output_view.shape[0], batch_size);

    EXPECT_EQ(output_view.shape[1], sequence_length);
    EXPECT_EQ(output_view.shape[2], output_embedding);
}
