#include "pch.h"

#include "../opennn/pooling_layer.h"
#include "../opennn/pooling_layer_3d.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/neural_network.h"

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
    const Index batch_size = parameters.input_data.dimension(0);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling>(
        parameters.input_shape,
        parameters.pool_dimensions,
        parameters.stride_shape,
        parameters.padding_dimensions,
        parameters.pooling_method,
        parameters.test_name));
    neural_network.compile();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(
        const_cast<type*>(parameters.input_data.data()),
        { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] }) };

    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
}

TEST_P(PoolingLayerTest, BackPropagate) {
    PoolingLayerConfig parameters = GetParam();
    const Index batch_size = parameters.input_data.dimension(0);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling>(
        parameters.input_shape,
        parameters.pool_dimensions,
        parameters.stride_shape,
        parameters.padding_dimensions,
        parameters.pooling_method,
        parameters.test_name));
    neural_network.compile();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(
        const_cast<type*>(parameters.input_data.data()),
        { batch_size, parameters.input_shape[0], parameters.input_shape[1], parameters.input_shape[2] }) };

    neural_network.forward_propagate(input_views, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
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
    const Index batch_size = 2;
    const Index seq = params.input_shape[0];
    const Index feat = params.input_shape[1];

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(params.input_shape, params.method, params.test_name));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}}, {{8, 7, 6, 5}, {4, 3, 2, 1}, {0, 0, 0, 0}}});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    EXPECT_EQ(output_view.shape[0], batch_size);
}

TEST_P(Pooling3dLayerTest, BackPropagate)
{
    Pooling3dConfig params = GetParam();
    const Index batch_size = 2;
    const Index seq = params.input_shape[0];
    const Index feat = params.input_shape[1];

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(params.input_shape, params.method, params.test_name));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 1, 1, 1}, {2, 2, 2, 2}, {0.5, 0.5, 0.5, 0.5}}, {{3, 3, 3, 3}, {1, 1, 1, 1}, {0, 0, 0, 0}}});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();
    EXPECT_EQ(output_view.shape[0], batch_size);
}
