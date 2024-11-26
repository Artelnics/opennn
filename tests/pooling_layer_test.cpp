#include "pch.h"

#include "../opennn/pooling_layer.h"
#include "../opennn/tensors.h"

Tensor<type, 4> generate_input_tensor_pooling(const Tensor<type, 2>& data,
                                              const vector<Index>& rows_indices,
                                              const vector<Index>& columns_indices,
                                              const dimensions& input_dimensions)
{ 
    Tensor<type, 4> input_tensor(rows_indices.size(),
                                 input_dimensions[0],
                                 input_dimensions[1],
                                 input_dimensions[2]);

    type* tensor_data = input_tensor.data();

    fill_tensor_data(data, rows_indices, columns_indices, tensor_data);
    
    return input_tensor;
}


struct PoolingLayerConfig {
    dimensions input_dimensions;
    dimensions pool_dimensions;
    dimensions stride_dimensions;
    dimensions padding_dimensions;
    PoolingLayer::PoolingMethod pooling_method;
    string test_name;
    Tensor<type, 4> input_data;
    Tensor<type, 4> expected_output;
};


class PoolingLayerTest : public ::testing::TestWithParam<PoolingLayerConfig> {};


INSTANTIATE_TEST_CASE_P(PoolingLayerTests, PoolingLayerTest, ::testing::Values(
    PoolingLayerConfig{
        {4, 4, 1}, {2, 2}, {2, 2}, {0, 0}, PoolingLayer::PoolingMethod::MaxPooling, "MaxPoolingNoPadding1Channel",
        ([] {
        Tensor<type, 2> data(4, 16);
        data.setValues({
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
            {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
            {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4},
            {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}
        });

        vector<Index> rows_indices = {0, 1, 2, 3};
        vector<Index> columns_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        return generate_input_tensor_pooling(data, rows_indices, columns_indices, {4, 4, 1});
    })(),
    ([] {
        Tensor<type, 4> expected_output(4, 2, 2, 1);
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
    }
    // More configurations here
    )
);


TEST_P(PoolingLayerTest, Constructor)
{
    PoolingLayerConfig parameters = GetParam();

    PoolingLayer pooling_layer(parameters.input_dimensions,
                               parameters.pool_dimensions,
                               parameters.stride_dimensions,
                               parameters.padding_dimensions,
                               parameters.pooling_method,
                               parameters.test_name);

    EXPECT_EQ(pooling_layer.get_type(), Layer::Type::Pooling);
    EXPECT_EQ(pooling_layer.get_input_dimensions(), parameters.input_dimensions);
    EXPECT_EQ(pooling_layer.get_pool_height(), parameters.pool_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_pool_width(), parameters.pool_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_row_stride(), parameters.stride_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_column_stride(), parameters.stride_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_padding_height(), parameters.padding_dimensions[0]);
    EXPECT_EQ(pooling_layer.get_padding_width(), parameters.padding_dimensions[1]);
    EXPECT_EQ(pooling_layer.get_pooling_method(), parameters.pooling_method);
}


TEST_P(PoolingLayerTest, ForwardPropagate) {

    PoolingLayerConfig parameters = GetParam();

    PoolingLayer pooling_layer(
        parameters.input_dimensions,
        parameters.pool_dimensions,
        parameters.stride_dimensions,
        parameters.padding_dimensions,
        parameters.pooling_method,
        parameters.test_name
    );
    
    const Index batch_samples_number = parameters.input_data.dimension(0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<PoolingLayerForwardPropagation>(batch_samples_number, &pooling_layer);

    pair<type*, dimensions> input_pair( parameters.input_data.data(),
        { batch_samples_number,
          parameters.input_dimensions[0],
          parameters.input_dimensions[1],
          parameters.input_dimensions[2] } );

    pooling_layer.forward_propagate({ input_pair }, forward_propagation, true);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    EXPECT_EQ(output_pair.second[0], batch_samples_number);
    EXPECT_EQ(output_pair.second[1], parameters.expected_output.dimension(1));
    EXPECT_EQ(output_pair.second[2], parameters.expected_output.dimension(2));
    EXPECT_EQ(output_pair.second[3], parameters.expected_output.dimension(3));

    TensorMap<Tensor<type, 4>> output_tensor(output_pair.first,
                                             batch_samples_number,
                                             parameters.expected_output.dimension(1),
                                             parameters.expected_output.dimension(2),
                                             parameters.expected_output.dimension(3));

    for (Index b = 0; b < batch_samples_number; ++b) {
        for (Index h = 0; h < parameters.expected_output.dimension(1); ++h) {
            for (Index w = 0; w < parameters.expected_output.dimension(2); ++w) {
                for (Index c = 0; c < parameters.expected_output.dimension(3); ++c) {
                    EXPECT_NEAR(output_tensor(b, h, w, c), parameters.expected_output(b, h, w, c), 1e-5)
                        << "Mismatch at batch=" << b << ", height=" << h
                        << ", width=" << w << ", channel=" << c;
                }
            }
        }
    }
}


TEST_P(PoolingLayerTest, BackPropagate) {

    PoolingLayerConfig parameters = GetParam();

    PoolingLayer pooling_layer(
        parameters.input_dimensions,
        parameters.pool_dimensions,
        parameters.stride_dimensions,
        parameters.padding_dimensions,
        parameters.pooling_method,
        parameters.test_name
    );

    const Index batch_samples_number = parameters.input_data.dimension(0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<PoolingLayerForwardPropagation>(batch_samples_number, &pooling_layer);

    unique_ptr<LayerBackPropagation> back_propagation =
        make_unique<PoolingLayerBackPropagation>(batch_samples_number, &pooling_layer);

    pair<type*, dimensions> input_pair( parameters.input_data.data(),
        { batch_samples_number,
          parameters.input_dimensions[0],
          parameters.input_dimensions[1],
          parameters.input_dimensions[2] } );

    pooling_layer.forward_propagate({ input_pair }, forward_propagation, true);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    pooling_layer.back_propagate({ input_pair }, { output_pair }, forward_propagation, back_propagation);

    vector<pair<type*, dimensions>> input_derivatives_pair = back_propagation.get()->get_input_derivative_pairs();

    EXPECT_EQ(input_derivatives_pair[0].second[0], batch_samples_number);
    EXPECT_EQ(input_derivatives_pair[0].second[1], parameters.input_data.dimension(1));
    EXPECT_EQ(input_derivatives_pair[0].second[2], parameters.input_data.dimension(2));
    EXPECT_EQ(input_derivatives_pair[0].second[3], parameters.input_data.dimension(3));

    // @todo check input derivatives vs numeric input_derivatives
}
