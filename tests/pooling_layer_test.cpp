#include "pch.h"

#include "../opennn/pooling_layer.h"
#include "../opennn/tensors.h"

using namespace opennn;

Tensor4 generate_input_tensor_pooling(const MatrixR& data,
                                      const vector<Index>& row_indices,
                                      const vector<Index>& column_indices,
                                      const Shape& input_shape)
{ 
    Tensor4 input_vector(row_indices.size(),
                                 input_shape[0],
                                 input_shape[1],
                                 input_shape[2]);

    type* tensor_data = input_vector.data();

    fill_tensor_data(data, row_indices, column_indices, tensor_data);
    
    return input_vector;
}


struct PoolingLayerConfig {
    Shape input_shape;
    Shape pool_dimensions;
    Shape stride_shape;
    Shape padding_dimensions;
    string pooling_method;
    string test_name;
    Tensor4 input_data;
    Tensor4 expected_output;
};


class PoolingLayerTest : public ::testing::TestWithParam<PoolingLayerConfig> {};


INSTANTIATE_TEST_CASE_P(PoolingLayerTests, PoolingLayerTest, ::testing::Values(
    PoolingLayerConfig
    {
        {4, 4, 1}, {2, 2}, {2, 2}, {0, 0}, "MaxPooling", "MaxPoolingNoPadding1Channel",
        ([] {
        MatrixR data(4, 16);
        data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3;

        const vector<Index> row_indices = {0, 1, 2, 3};
        const vector<Index> column_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        return generate_input_tensor_pooling(data, row_indices, column_indices, {4, 4, 1});
        })(),
        ([] {
            Tensor4 expected_output(4, 2, 2, 1);
            expected_output.setValues({
                                      {{{6}, {14}},
                                       {{8}, {16}}},

                                      {{{16}, {8}},
                                       {{14}, {6}}},

                                      {{{2}, {4}},
                                       {{2}, {4}}},

                                      {{{1}, {3}},
                                       {{1}, {3}}}
                                      });
            return expected_output;
        })()
    },
    PoolingLayerConfig
    {
        {4, 4, 1}, {2, 2}, {2, 2}, {0, 0}, "AveragePooling", "AveragePoolingNoPadding1Channel",
        ([] {
        MatrixR data(4, 16);
        data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3;

        const vector<Index> row_indices = {0, 1, 2, 3};
        const vector<Index> column_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        return generate_input_tensor_pooling(data, row_indices, column_indices, {4, 4, 1});
        })(),
        ([] {
            Tensor4 expected_output(4, 2, 2, 1);
            expected_output.setValues({
                                        {{{3.5}, {11.5}},
                                        {{5.5}, {13.5}}},

                                        {{{13.5}, {5.5}},
                                        {{11.5}, {3.5}}},

                                        {{{1.5}, {3.5}},
                                        {{1.5}, {3.5}}},

                                        {{{0.5}, {2.5}},
                                        {{0.5}, {2.5}}}
                                        });
            return expected_output;
        })()
    }
    // More configurations here
    )
);


TEST_P(PoolingLayerTest, Constructor)
{
    PoolingLayerConfig parameters = GetParam();

    Pooling pooling_layer(parameters.input_shape,
                               parameters.pool_dimensions,
                               parameters.stride_shape,
                               parameters.padding_dimensions,
                               parameters.pooling_method,
                               parameters.test_name);

    EXPECT_EQ(pooling_layer.get_name(), "Pooling");
    EXPECT_EQ(pooling_layer.get_input_shape(), parameters.input_shape);
    EXPECT_EQ(pooling_layer.get_pool_height(), parameters.pool_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_pool_width(), parameters.pool_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_row_stride(), parameters.stride_shape[0]);
    EXPECT_EQ(pooling_layer.get_column_stride(), parameters.stride_shape[1]);
    EXPECT_EQ(pooling_layer.get_padding_height(), parameters.padding_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_padding_width(), parameters.padding_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_pooling_method(), parameters.pooling_method);
}


TEST_P(PoolingLayerTest, ForwardPropagate)
{
    PoolingLayerConfig parameters = GetParam();

    Pooling pooling_layer(
        parameters.input_shape,
        parameters.pool_dimensions,
        parameters.stride_shape,
        parameters.padding_dimensions,
        parameters.pooling_method,
        parameters.test_name
    );

    const Index batch_size = parameters.input_data.dimension(0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<PoolingForwardPropagation>(batch_size, &pooling_layer);

    Tensor1 workspace(get_size(forward_propagation->get_workspace_views()));
    link(workspace.data(), forward_propagation->get_workspace_views());

    TensorView input_view( parameters.input_data.data(),
        { batch_size,
          parameters.input_shape[0],
          parameters.input_shape[1],
          parameters.input_shape[2] } );

    pooling_layer.forward_propagate({ input_view }, forward_propagation, false);

    TensorView output_pair = forward_propagation->get_outputs();

    EXPECT_EQ(output_pair.shape[0], batch_size);
    EXPECT_EQ(output_pair.shape[1], parameters.expected_output.dimension(1));
    EXPECT_EQ(output_pair.shape[2], parameters.expected_output.dimension(2));
    EXPECT_EQ(output_pair.shape[3], parameters.expected_output.dimension(3));

    TensorMap4 output_tensor = tensor_map<4>(output_pair);

    for (Index b = 0; b < batch_size; ++b) {
        for (Index h = 0; h < parameters.expected_output.dimension(1); ++h) {
            for (Index w = 0; w < parameters.expected_output.dimension(2); ++w) {
                for (Index c = 0; c < parameters.expected_output.dimension(3); ++c) {
                    EXPECT_NEAR(output_tensor(b, h, w, c), parameters.expected_output(b, h, w, c), 1e-5);
                }
            }
        }
    }
}


TEST_P(PoolingLayerTest, BackPropagate) {

    PoolingLayerConfig parameters = GetParam();

    Pooling pooling_layer(
        parameters.input_shape,
        parameters.pool_dimensions,
        parameters.stride_shape,
        parameters.padding_dimensions,
        parameters.pooling_method,
        parameters.test_name
    );

    const Index batch_size = parameters.input_data.dimension(0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<PoolingForwardPropagation>(batch_size, &pooling_layer);

    unique_ptr<LayerBackPropagation> back_propagation =
        make_unique<PoolingBackPropagation>(batch_size, &pooling_layer);

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    TensorView input_view( parameters.input_data.data(),
        { batch_size,
          parameters.input_shape[0],
          parameters.input_shape[1],
          parameters.input_shape[2] } );

    pooling_layer.forward_propagate({ input_view }, forward_propagation, true);

    TensorView output_pair = forward_propagation->get_outputs();

    pooling_layer.back_propagate({ input_view }, { output_pair }, forward_propagation, back_propagation);

    vector<TensorView> input_derivatives_pair = back_propagation->get_input_gradients();

    EXPECT_EQ(input_derivatives_pair[0].shape[0], batch_size);
    EXPECT_EQ(input_derivatives_pair[0].shape[1], parameters.input_data.dimension(1));
    EXPECT_EQ(input_derivatives_pair[0].shape[2], parameters.input_data.dimension(2));
    EXPECT_EQ(input_derivatives_pair[0].shape[3], parameters.input_data.dimension(3));
}
