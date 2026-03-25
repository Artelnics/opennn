#include "pch.h"

#include "../opennn/pooling_layer.h"
#include "../opennn/pooling_layer_3d.h"
#include "../opennn/tensor_utilities.h"

using namespace opennn;

// POOLING 4D (IMAGES)

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
                                                                          expected_output.setValues({{{{6}, {8}},{{14}, {16}}},{{{16}, {14}},{{8}, {6}}},{{{2}, {2}},{{4}, {4}}},{{{1}, {1}},{{3}, {3}}}});
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
                                                                          expected_output.setValues({{{{3.5}, {5.5}},{{11.5}, {13.5}}},{{{13.5}, {11.5}},{{5.5}, {3.5}}},{{{1.5}, {1.5}},{{3.5}, {3.5}}},{{{0.5}, {0.5}},{{2.5}, {2.5}}}});
                                                                          return expected_output;
                                                                      })()
                                                                  }
                                                                  ));

TEST_P(PoolingLayerTest, Constructor)
{
    PoolingLayerConfig parameters = GetParam();
    Pooling pooling_layer(parameters.input_shape, parameters.pool_dimensions, parameters.stride_shape, parameters.padding_dimensions, parameters.pooling_method, parameters.test_name);
    EXPECT_EQ(pooling_layer.get_name(), "Pooling");
    EXPECT_EQ(pooling_layer.get_input_shape(), parameters.input_shape);
}

TEST_P(PoolingLayerTest, ForwardPropagate)
{
    PoolingLayerConfig parameters = GetParam();
    Pooling pooling_layer(parameters.input_shape, parameters.pool_dimensions, parameters.stride_shape, parameters.padding_dimensions, parameters.pooling_method, parameters.test_name);
    const Index batch_size = parameters.input_data.dimension(0);
    unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<PoolingForwardPropagation>(batch_size, &pooling_layer);
    forward_propagation->initialize();
    Tensor1 workspace(get_size(forward_propagation->get_workspace_views()));
    link(workspace.data(), forward_propagation->get_workspace_views());
    forward_propagation->inputs = { TensorView(parameters.input_data.data(), { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] }) };
    pooling_layer.forward_propagate(forward_propagation, false);
    TensorMap4 output_tensor = tensor_map<4>(forward_propagation->get_outputs());
    for (Index b = 0; b < batch_size; ++b)
        for (Index h = 0; h < parameters.expected_output.dimension(1); ++h)
            for (Index w = 0; w < parameters.expected_output.dimension(2); ++w)
                for (Index c = 0; c < parameters.expected_output.dimension(3); ++c)
                    EXPECT_NEAR(output_tensor(b, h, w, c), parameters.expected_output(b, h, w, c), 1e-5);
#ifdef OPENNN_CUDA
    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda = make_unique<PoolingForwardPropagationCuda>(batch_size, &pooling_layer);
    forward_propagation_cuda->initialize();
    vector<TensorViewCuda*> workspace_views_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_device({get_size(workspace_views_device)});
    link(layer_workspace_device.data, workspace_views_device);
    TensorCuda inputs_device({ batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] });
    CHECK_CUDA(cudaMemcpy(inputs_device.data, parameters.input_data.data(), parameters.input_data.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda->inputs = { inputs_device.view() };
    pooling_layer.forward_propagate(forward_propagation_cuda, false);
    vector<type> host_output_from_gpu(forward_propagation_cuda->outputs.size());
    CHECK_CUDA(cudaMemcpy(host_output_from_gpu.data(), forward_propagation_cuda->outputs.data, host_output_from_gpu.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < forward_propagation->get_outputs().size(); ++i)
        EXPECT_NEAR(forward_propagation->get_outputs().data[i], host_output_from_gpu[i], 1e-5);
#endif
}

TEST_P(PoolingLayerTest, BackPropagate) {
    PoolingLayerConfig parameters = GetParam();
    Pooling pooling_layer(parameters.input_shape, parameters.pool_dimensions, parameters.stride_shape, parameters.padding_dimensions, parameters.pooling_method, parameters.test_name);
    const Index batch_size = parameters.input_data.dimension(0);
    unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<PoolingForwardPropagation>(batch_size, &pooling_layer);
    forward_propagation->initialize();
    Tensor1 workspace_fw(get_size(forward_propagation->get_workspace_views()));
    link(workspace_fw.data(), forward_propagation->get_workspace_views());
    forward_propagation->inputs = { TensorView(parameters.input_data.data(), { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] }) };
    pooling_layer.forward_propagate(forward_propagation, true);
    unique_ptr<LayerBackPropagation> back_propagation = make_unique<PoolingBackPropagation>(batch_size, &pooling_layer);
    back_propagation->initialize();
    vector<TensorView*> bp_workspace_views = back_propagation->get_workspace_views();
    Tensor1 workspace_bw(get_size(bp_workspace_views));
    link(workspace_bw.data(), bp_workspace_views);
    Tensor1 deltas(forward_propagation->get_outputs().size());
    deltas.setConstant(1.0);
    back_propagation->output_gradients = { TensorView(deltas.data(), forward_propagation->get_outputs().shape) };
    pooling_layer.back_propagate(forward_propagation, back_propagation);
#ifdef OPENNN_CUDA
    unique_ptr<LayerForwardPropagationCuda> forward_propagation_cuda = make_unique<PoolingForwardPropagationCuda>(batch_size, &pooling_layer);
    forward_propagation_cuda->initialize();
    vector<TensorViewCuda*> workspace_fw_device = forward_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_fw_device({get_size(workspace_fw_device)});
    link(layer_workspace_fw_device.data, workspace_fw_device);
    TensorCuda inputs_device({ batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] });
    CHECK_CUDA(cudaMemcpy(inputs_device.data, parameters.input_data.data(), parameters.input_data.size() * sizeof(type), cudaMemcpyHostToDevice));
    forward_propagation_cuda->inputs = { inputs_device.view() };
    pooling_layer.forward_propagate(forward_propagation_cuda, true);
    unique_ptr<LayerBackPropagationCuda> back_propagation_cuda = make_unique<PoolingBackPropagationCuda>(batch_size, &pooling_layer);
    back_propagation_cuda->initialize();
    vector<TensorViewCuda*> bp_workspace_device = back_propagation_cuda->get_workspace_views();
    TensorCuda layer_workspace_bw_device({get_size(bp_workspace_device)});
    link(layer_workspace_bw_device.data, bp_workspace_device);
    TensorCuda delta_device({deltas.size()});
    CHECK_CUDA(cudaMemcpy(delta_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));
    back_propagation_cuda->output_gradients = { TensorViewCuda(delta_device.data, forward_propagation_cuda->outputs.descriptor_handle) };
    pooling_layer.back_propagate(forward_propagation_cuda, back_propagation_cuda);
    vector<type> host_input_grads(back_propagation_cuda->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), back_propagation_cuda->input_gradients[0].data, host_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < back_propagation->get_input_gradients()[0].size(); ++i)
        EXPECT_NEAR(back_propagation->get_input_gradients()[0].data[i], host_input_grads[i], 1e-5);
#endif
}

// POOLING 3D (TRANSFORMERS)

struct Pooling3dConfig {
    Shape input_shape; // {seq, features}
    Pooling3d::PoolingMethod method;
    string test_name;
};

class Pooling3dLayerTest : public ::testing::TestWithParam<Pooling3dConfig> {};

INSTANTIATE_TEST_SUITE_P(Pooling3dLayerTests, Pooling3dLayerTest, ::testing::Values(
                                                                      Pooling3dConfig{{3, 4}, Pooling3d::PoolingMethod::MaxPooling, "MaxPooling3d"},
                                                                      Pooling3dConfig{{3, 4}, Pooling3d::PoolingMethod::AveragePooling, "AveragePooling3d"}
                                                                      ));

TEST_P(Pooling3dLayerTest, ForwardPropagate)
{
    Pooling3dConfig params = GetParam();
    Pooling3d pooling_layer(params.input_shape, params.method, params.test_name);
    const Index batch_size = 2;
    const Index seq = params.input_shape[0];
    const Index feat = params.input_shape[1];

    unique_ptr<LayerForwardPropagation> fw = make_unique<Pooling3dForwardPropagation>(batch_size, &pooling_layer);
    fw->initialize();
    Tensor1 workspace(get_size(fw->get_workspace_views()));
    link(workspace.data(), fw->get_workspace_views());

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}}, {{8, 7, 6, 5}, {4, 3, 2, 1}, {0, 0, 0, 0}}});
    fw->inputs = { TensorView(input_data.data(), {batch_size, seq, feat}) };

    pooling_layer.forward_propagate(fw, false);

#ifdef OPENNN_CUDA
    unique_ptr<LayerForwardPropagationCuda> fw_cuda = make_unique<Pooling3dForwardPropagationCuda>(batch_size, &pooling_layer);
    fw_cuda->initialize();

    vector<TensorViewCuda*> ws_views_device = fw_cuda->get_workspace_views();
    TensorCuda ws_device({get_size(ws_views_device)});
    link(ws_device.data, ws_views_device);

    TensorCuda inputs_device({batch_size, seq, feat});
    CHECK_CUDA(cudaMemcpy(inputs_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));
    fw_cuda->inputs = { inputs_device.view() };

    pooling_layer.forward_propagate(fw_cuda, false);

    vector<type> host_gpu(fw_cuda->outputs.size());
    CHECK_CUDA(cudaMemcpy(host_gpu.data(), fw_cuda->outputs.data, host_gpu.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for(Index i = 0; i < fw->get_outputs().size(); ++i) EXPECT_NEAR(fw->get_outputs().data[i], host_gpu[i], 1e-5);
#endif
}

TEST_P(Pooling3dLayerTest, BackPropagate)
{
    Pooling3dConfig params = GetParam();
    Pooling3d pooling_layer(params.input_shape, params.method, params.test_name);
    const Index batch_size = 2;
    const Index seq = params.input_shape[0];
    const Index feat = params.input_shape[1];

    unique_ptr<LayerForwardPropagation> fw = make_unique<Pooling3dForwardPropagation>(batch_size, &pooling_layer);
    fw->initialize();
    Tensor1 workspace_fw(get_size(fw->get_workspace_views()));
    link(workspace_fw.data(), fw->get_workspace_views());

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 1, 1, 1}, {2, 2, 2, 2}, {0.5, 0.5, 0.5, 0.5}}, {{3, 3, 3, 3}, {1, 1, 1, 1}, {0, 0, 0, 0}}});
    fw->inputs = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    pooling_layer.forward_propagate(fw, true);

    unique_ptr<LayerBackPropagation> bw = make_unique<Pooling3dBackPropagation>(batch_size, &pooling_layer);
    bw->initialize();
    Tensor1 workspace_bw(get_size(bw->get_workspace_views()));
    link(workspace_bw.data(), bw->get_workspace_views());

    MatrixR deltas(batch_size, feat);
    deltas.setConstant(1.0);
    bw->output_gradients = { TensorView(deltas.data(), {batch_size, feat}) };
    pooling_layer.back_propagate(fw, bw);

#ifdef OPENNN_CUDA
    unique_ptr<LayerForwardPropagationCuda> fw_cuda = make_unique<Pooling3dForwardPropagationCuda>(batch_size, &pooling_layer);
    fw_cuda->initialize();
    vector<TensorViewCuda*> ws_fw_device = fw_cuda->get_workspace_views();
    TensorCuda ws_fw_mem({get_size(ws_fw_device)});
    link(ws_fw_mem.data, ws_fw_device);

    TensorCuda inputs_device({batch_size, seq, feat});
    CHECK_CUDA(cudaMemcpy(inputs_device.data, input_data.data(), input_data.size() * sizeof(type), cudaMemcpyHostToDevice));
    fw_cuda->inputs = { inputs_device.view() };
    pooling_layer.forward_propagate(fw_cuda, true);

    unique_ptr<LayerBackPropagationCuda> bw_cuda = make_unique<Pooling3dBackPropagationCuda>(batch_size, &pooling_layer);
    bw_cuda->initialize();

    vector<TensorViewCuda*> ws_bw_device = bw_cuda->get_workspace_views();
    TensorCuda ws_bw_mem({get_size(ws_bw_device)});
    link(ws_bw_mem.data, ws_bw_device);

    TensorCuda deltas_device({batch_size, feat});
    CHECK_CUDA(cudaMemcpy(deltas_device.data, deltas.data(), deltas.size() * sizeof(type), cudaMemcpyHostToDevice));
    bw_cuda->output_gradients = { deltas_device.view() };

    pooling_layer.back_propagate(fw_cuda, bw_cuda);

    vector<type> host_input_grads(bw_cuda->input_gradients[0].size());
    CHECK_CUDA(cudaMemcpy(host_input_grads.data(), bw_cuda->input_gradients[0].data, host_input_grads.size() * sizeof(type), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < bw->get_input_gradients()[0].size(); ++i)
        EXPECT_NEAR(bw->get_input_gradients()[0].data[i], host_input_grads[i], 1e-5);
#endif
}
