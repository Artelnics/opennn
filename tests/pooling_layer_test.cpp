#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/pooling_layer.h"
#include "opennn/pooling_layer_3d.h"
#include "opennn/tensor_types.h"
#include "opennn/statistics.h"
#include "opennn/dense_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

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
    EXPECT_EQ(pooling_layer.get_label(), parameters.test_name);
    EXPECT_EQ(pooling_layer.get_input_shape(), parameters.input_shape);
    EXPECT_EQ(pooling_layer.get_pooling_method(), string_to_pooling_method(parameters.pooling_method));

    const Index expected_output_height = (parameters.input_shape[0] - parameters.pool_dimensions[0] + 2 * parameters.padding_dimensions[0]) / parameters.stride_shape[0] + 1;
    const Index expected_output_width  = (parameters.input_shape[1] - parameters.pool_dimensions[1] + 2 * parameters.padding_dimensions[1]) / parameters.stride_shape[1] + 1;

    EXPECT_EQ(pooling_layer.get_output_shape(), (Shape{expected_output_height, expected_output_width, parameters.input_shape[2]}));
}

TEST(PoolingLayerTest, DefaultConstructorDerivedShapes)
{
    Pooling pooling_layer;

    EXPECT_EQ(pooling_layer.get_name(), "Pooling");
    EXPECT_EQ(pooling_layer.get_input_shape(), (Shape{2, 2, 1}));
    EXPECT_EQ(pooling_layer.get_pooling_method(), PoolingMethod::MaxPooling);

    EXPECT_EQ(pooling_layer.get_output_shape(), (Shape{1, 1, 1}));
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
    ASSERT_EQ(output_view.size(), parameters.expected_output.size());

    const float* output_data = output_view.as<type>();

    for (Index i = 0; i < parameters.expected_output.size(); ++i)
        EXPECT_NEAR(output_data[i], parameters.expected_output.data()[i], 1.0e-4f);
}

TEST(PoolingLayerTest, StridePaddingForwardValues)
{
    const Index batch_size = 1;
    const Shape input_shape{3, 3, 1};
    const Shape pool_dimensions{2, 2};
    const Shape stride_shape{1, 1};
    const Shape padding_dimensions{0, 0};

    Tensor4 input_data(batch_size, 3, 3, 1);
    input_data.setValues({{{{1}, {2}, {3}}, {{4}, {5}, {6}}, {{7}, {8}, {9}}}});

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling>(
        input_shape, pool_dimensions, stride_shape, padding_dimensions, "MaxPooling", "overlap_max"));
    neural_network.compile();

    EXPECT_EQ(neural_network.get_layer(0)->get_output_shape(), (Shape{2, 2, 1}));

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, 3, 3, 1}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    ASSERT_EQ(output_view.size(), 4);

    const float* max_data = output_view.as<type>();
    const float expected_max[4] = {5, 6, 8, 9};
    for (Index i = 0; i < 4; ++i)
        EXPECT_NEAR(max_data[i], expected_max[i], 1.0e-4f);

    NeuralNetwork average_network;
    average_network.add_layer(make_unique<Pooling>(
        input_shape, pool_dimensions, stride_shape, padding_dimensions, "AveragePooling", "overlap_average"));
    average_network.compile();

    ForwardPropagation average_forward(batch_size, &average_network);
    vector<TensorView> average_views = { TensorView(input_data.data(), {batch_size, 3, 3, 1}) };
    average_network.forward_propagate(average_views, average_forward, false);

    TensorView average_view = average_forward.get_outputs();
    ASSERT_EQ(average_view.size(), 4);

    const float* average_data = average_view.as<type>();
    const float expected_average[4] = {3, 4, 6, 7};
    for (Index i = 0; i < 4; ++i)
        EXPECT_NEAR(average_data[i], expected_average[i], 1.0e-4f);
}


struct Pooling3dConfig {
    Shape input_shape;
    PoolingMethod method;
    string test_name;
};

class Pooling3dLayerTest : public ::testing::TestWithParam<Pooling3dConfig> {};

INSTANTIATE_TEST_SUITE_P(Pooling3dLayerTests, Pooling3dLayerTest, ::testing::Values(
                                                                      Pooling3dConfig{{3, 4}, PoolingMethod::MaxPooling, "MaxPooling3d"},
                                                                      Pooling3dConfig{{3, 4}, PoolingMethod::AveragePooling, "AveragePooling3d"}
                                                                      ));

TEST_P(Pooling3dLayerTest, Constructor)
{
    Pooling3dConfig params = GetParam();
    Pooling3d pooling_layer(params.input_shape, params.method, params.test_name);

    EXPECT_EQ(pooling_layer.get_name(), "Pooling3d");
    EXPECT_EQ(pooling_layer.get_label(), params.test_name);
    EXPECT_EQ(pooling_layer.get_input_shape(), params.input_shape);
    EXPECT_EQ(pooling_layer.get_pooling_method(), params.method);

    EXPECT_EQ(pooling_layer.get_output_shape(), (Shape{params.input_shape[1]}));
}

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
    ASSERT_EQ(output_view.shape[0], batch_size);
    ASSERT_EQ(output_view.size(), batch_size * feat);

    const float* output_data = output_view.as<type>();

    MatrixR expected(batch_size, feat);
    if (params.method == PoolingMethod::MaxPooling)
        expected << 5, 6, 7, 8,
                    8, 7, 6, 5;
    else
        expected << 8.0f / 3.0f, 10.0f / 3.0f, 12.0f / 3.0f, 14.0f / 3.0f,
                    6, 5, 4, 3;

    for (Index sample = 0; sample < batch_size; ++sample)
        for (Index feature = 0; feature < feat; ++feature)
            EXPECT_NEAR(output_data[sample * feat + feature], expected(sample, feature), 1.0e-4f);
}

TEST_P(Pooling3dLayerTest, Pooling3dBackwardGradientMatchesNumerical)
{
    Pooling3dConfig params = GetParam();

    const Index samples_number = 5;
    const Index seq = params.input_shape[0];
    const Index feat = params.input_shape[1];
    const Index targets_number = 2;

    const Shape input_shape{seq, feat};

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<opennn::Dense>(input_shape, Shape{feat}, "Identity"));
    neural_network.add_layer(make_unique<Pooling3d>(neural_network.get_output_shape(), params.method, params.test_name));
    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
