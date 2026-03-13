#include "pch.h"

#include "../opennn/pooling_layer.h"
#include "../opennn/tensor_utilities.h"

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

INSTANTIATE_TEST_SUITE_P(PoolingLayerTests, PoolingLayerTest, ::testing::Values(
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
                                                                              {{{6}, {8}},
                                                                               {{14}, {16}}},

<<<<<<< Updated upstream
                                                                              {{{16}, {14}},
                                                                               {{8}, {6}}},

                                                                              {{{2}, {2}},
                                                                               {{4}, {4}}},

                                                                              {{{1}, {1}},
                                                                               {{3}, {3}}}
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
=======
        return generate_input_tensor_pooling(data, row_indices, column_indices, {4, 4, 1});
        })(),
        ([] {
            Tensor4 expected_output(4, 2, 2, 1);
            expected_output.setValues({
                                        {{{6}, {8}},
                                        {{14}, {16}}},

                                        {{{16}, {14}},
                                        {{8}, {6}}},

                                        {{{2}, {2}},
                                        {{4}, {4}}},

                                        {{{1}, {1}},
                                        {{3}, {3}}}
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
>>>>>>> Stashed changes

                                                                          return generate_input_tensor_pooling(data, row_indices, column_indices, {4, 4, 1});
                                                                      })(),
                                                                      ([] {
                                                                          Tensor4 expected_output(4, 2, 2, 1);
                                                                          expected_output.setValues({
                                                                              {{{3.5}, {5.5}},
                                                                               {{11.5}, {13.5}}},

<<<<<<< Updated upstream
                                                                              {{{13.5}, {11.5}},
                                                                               {{5.5}, {3.5}}},

                                                                              {{{1.5}, {1.5}},
                                                                               {{3.5}, {3.5}}},
=======
        return generate_input_tensor_pooling(data, row_indices, column_indices, {4, 4, 1});
        })(),
        ([] {
            Tensor4 expected_output(4, 2, 2, 1);
            expected_output.setValues({
                                        {{{3.5}, {5.5}},
                                        {{11.5}, {13.5}}},

                                        {{{13.5}, {11.5}},
                                        {{5.5}, {3.5}}},

                                        {{{1.5}, {1.5}},
                                        {{3.5}, {3.5}}},

                                        {{{0.5}, {0.5}},
                                        {{2.5}, {2.5}}}
                                    });
            return expected_output;
        })()
    }
    // More configurations here
    )
);
>>>>>>> Stashed changes

                                                                              {{{0.5}, {0.5}},
                                                                               {{2.5}, {2.5}}}
                                                                          });
                                                                          return expected_output;
                                                                      })()
                                                                  }
                                                                  ));

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

    forward_propagation->initialize();

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

#ifdef OPENNN_CUDA
    TensorCuda input_data_device({batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2]});
    CHECK_CUDA(cudaMemcpy(input_data_device.data, parameters.input_data.data(), input_data_device.size() * sizeof(type), cudaMemcpyHostToDevice));
    vector<TensorViewCuda> input_views_device = { input_data_device.view() };

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda =
        make_unique<PoolingForwardPropagationCuda>(batch_size, &pooling_layer);
    forward_propagation_cuda->initialize();

    vector<TensorViewCuda*> workspace_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);

    pooling_layer.forward_propagate(input_views_device, forward_propagation_cuda, false);

    TensorViewCuda output_view_device = forward_propagation_cuda->outputs;
    EXPECT_EQ(output_view_device.size(), output_pair.size());

    vector<type> host_output_from_gpu(output_view_device.size());
    CHECK_CUDA(cudaMemcpy(host_output_from_gpu.data(), output_view_device.data, output_view_device.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < output_pair.size(); ++i)
        EXPECT_NEAR(output_pair.data[i], host_output_from_gpu[i], 1e-5);

#endif
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
    forward_propagation->initialize();

    unique_ptr<LayerBackPropagation> back_propagation =
        make_unique<PoolingBackPropagation>(batch_size, &pooling_layer);
    back_propagation->initialize();

    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    link(workspace_fw.data(), forward_propagation->get_workspace_views());

    TensorView input_view( parameters.input_data.data(),
                          { batch_size,
                           parameters.input_shape[0],
                           parameters.input_shape[1],
                           parameters.input_shape[2] } );

    pooling_layer.forward_propagate({ input_view }, forward_propagation, true);

    TensorView output_pair = forward_propagation->get_outputs();

    Tensor1 deltas(output_pair.size());
    deltas.setConstant(1.0);
    TensorView delta_view(deltas.data(), output_pair.shape);

    pooling_layer.back_propagate({ input_view }, { delta_view }, forward_propagation, back_propagation);

    vector<TensorView> input_derivatives_pair = back_propagation->get_input_gradients();

    EXPECT_EQ(input_derivatives_pair[0].shape[0], batch_size);
    EXPECT_EQ(input_derivatives_pair[0].shape[1], parameters.input_data.dimension(1));
    EXPECT_EQ(input_derivatives_pair[0].shape[2], parameters.input_data.dimension(2));
    EXPECT_EQ(input_derivatives_pair[0].shape[3], parameters.input_data.dimension(3));

#ifdef OPENNN_CUDA
    TensorCuda input_data_device({batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2]});
    CHECK_CUDA(cudaMemcpy(input_data_device.data, parameters.input_data.data(), input_data_device.size() * sizeof(type), cudaMemcpyHostToDevice));
    vector<TensorViewCuda> input_views_device = { input_data_device.view() };

    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda =
        make_unique<PoolingForwardPropagationCuda>(batch_size, &pooling_layer);
    forward_propagation_cuda->initialize();

    vector<TensorViewCuda*> workspace_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);

    pooling_layer.forward_propagate(input_views_device, forward_propagation_cuda, true);

    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda =
        make_unique<PoolingBackPropagationCuda>(batch_size, &pooling_layer);
    back_propagation_cuda->initialize();

    TensorCuda delta_device({output_pair.shape[0], output_pair.shape[1], output_pair.shape[2], output_pair.shape[3]});
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), delta_device.size() * sizeof(type), cudaMemcpyHostToDevice));
    vector<TensorViewCuda> delta_views_device = { delta_device.view() };

    pooling_layer.back_propagate(input_views_device, delta_views_device, forward_propagation_cuda, back_propagation_cuda);

    vector<type> host_input_grads(back_propagation_cuda->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), back_propagation_cuda->input_gradients[0].data, back_propagation_cuda->input_gradients[0].size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < input_derivatives_pair[0].size(); ++i)
        EXPECT_NEAR(input_derivatives_pair[0].data[i], host_input_grads[i], 1e-5);

#endif
}
