#include "pch.h"

#include "../opennn/flatten_layer.h"

using namespace opennn;

class FlattenLayerTest : public ::testing::Test
{
protected:
    const Index height = 4;
    const Index width = 4;
    const Index channels = 3;
    const dimensions input_dimensions = { height, width, channels };

    unique_ptr<Flatten<4>> flatten_layer;

    void SetUp() override
    {
        flatten_layer = make_unique<Flatten<4>>(input_dimensions);
    }
};


TEST_F(FlattenLayerTest, Constructor)
{
    EXPECT_EQ(flatten_layer->get_input_dimensions(), input_dimensions);
    EXPECT_EQ(flatten_layer->get_output_dimensions(), dimensions{ height * width * channels });
    EXPECT_EQ(flatten_layer->get_name(), "Flatten4d");
}


TEST_F(FlattenLayerTest, ForwardPropagate)
{
    const Index batch_size = 2;

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<FlattenForwardPropagation<2>>(batch_size, flatten_layer.get());

    Tensor<type, 4> inputs(batch_size, height, width, channels);
    inputs.setConstant(1.23f);

    auto eigen_dims = inputs.dimensions();
    dimensions dims_vector(eigen_dims.begin(), eigen_dims.end());
    TensorView input_view(inputs.data(), dims_vector);

    flatten_layer->forward_propagate({ input_view }, forward_propagation, false);

    const TensorView output_pair = forward_propagation->get_output_pair();
    const dimensions& output_dims = output_pair.dims;

    ASSERT_EQ(output_dims.size(), 2) << "Flatten<4> should produce a 2D tensor (batch, features).";
    EXPECT_EQ(output_dims[0], batch_size);
    EXPECT_EQ(output_dims[1], height * width * channels);
}


TEST_F(FlattenLayerTest, BackPropagate)
{
    /*
    const Index batch_size = 2;

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<FlattenForwardPropagation<2>>(batch_size, flatten_layer.get());
    unique_ptr<LayerBackPropagation> back_propagation =
        make_unique<FlattenBackPropagation<2>>(batch_size, flatten_layer.get());

    Tensor<type, 4> inputs(batch_size, height, width, channels);
    inputs.setRandom();

    TensorView input_view(inputs.data(), dimensions{ batch_size, height, width, channels });

    const Index flattened_size = height * width * channels;
    Tensor<type, 2> output_derivatives(batch_size, flattened_size);
    output_derivatives.setConstant(1.0f);
    TensorView output_derivatives_view(output_derivatives.data(), dimensions{ batch_size, flattened_size });

    flatten_layer->forward_propagate({ input_view }, forward_propagation, true);

    flatten_layer->back_propagate({ input_view }, { output_derivatives_view }, forward_propagation, back_propagation);

    const vector<TensorView> input_derivative_views = back_propagation->get_input_derivative_views();

    ASSERT_EQ(input_derivative_views.size(), 1) << "Flatten layer should have one input derivative.";

    const TensorView& input_derivative_view = input_derivative_views[0];

    ASSERT_EQ(input_derivative_view.rank(), 4) << "Input derivative rank should be 4.";
    EXPECT_EQ(input_derivative_view.dims[0], batch_size);
    EXPECT_EQ(input_derivative_view.dims[1], height);
    EXPECT_EQ(input_derivative_view.dims[2], width);
    EXPECT_EQ(input_derivative_view.dims[3], channels);

    TensorMap<const Tensor<const type, 4>> input_derivatives_map(input_derivative_view.data,
        input_derivative_view.dims[0],
        input_derivative_view.dims[1],
        input_derivative_view.dims[2],
        input_derivative_view.dims[3]);

    const type tolerance = 1e-7f;
    bool all_values_are_correct = true;
    for (Index i = 0; i < input_derivatives_map.size(); ++i)
    {
        if (abs(input_derivatives_map.data()[i] - 1.0f) > tolerance)
        {
            all_values_are_correct = false;
            break;
        }
    }

    EXPECT_TRUE(all_values_are_correct) << "All values in the input derivative tensor should be 1.0.";
    */
}
