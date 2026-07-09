#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/pooling_layer.h"
#include "opennn/pooling_layer_3d.h"
#include "opennn/dense_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

TEST(Pool3dOperatoreratorTest, MaxConstructorShapes)
{
    const Shape input_shape{3, 4};
    Pooling3d layer(input_shape, PoolingMethod::MaxPooling, "max_pool_3d");

    EXPECT_EQ(layer.get_name(), "Pooling3d");
    EXPECT_EQ(layer.get_input_shape(), input_shape);
    EXPECT_EQ(layer.get_sequence_length(), 3);
    EXPECT_EQ(layer.get_input_features(), 4);
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::MaxPooling);
    EXPECT_EQ(layer.get_output_shape(), Shape{4});
}

TEST(Pool3dOperatoreratorTest, AverageConstructorShapes)
{
    const Shape input_shape{5, 2};
    Pooling3d layer(input_shape, PoolingMethod::AveragePooling, "avg_pool_3d");

    EXPECT_EQ(layer.get_input_features(), 2);
    EXPECT_EQ(layer.get_sequence_length(), 5);
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::AveragePooling);
    EXPECT_EQ(layer.get_output_shape(), Shape{2});
}

TEST(Pool3dOperatoreratorTest, SetPoolingMethodFromString)
{
    Pooling3d layer({3, 4}, PoolingMethod::MaxPooling, "pool");
    layer.set_pooling_method("AveragePooling");
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::AveragePooling);
    layer.set_pooling_method("MaxPooling");
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::MaxPooling);
}

TEST(Pool3dOperatoreratorTest, ForwardMaxValuesAndShape)
{
    const Index batch_size = 2;
    const Index seq = 3;
    const Index feat = 4;
    const Shape input_shape{seq, feat};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(input_shape, PoolingMethod::MaxPooling, "max_pool"));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}},
                          {{8, 7, 6, 5}, {4, 3, 2, 1}, {0, 0, 0, 0}}});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], feat);

    const float expected[8] = {5, 6, 7, 8, 8, 7, 6, 5};
    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], expected[i], 1e-6f);
}

TEST(Pool3dOperatoreratorTest, ForwardAverageValuesWithPadding)
{
    const Index batch_size = 2;
    const Index seq = 3;
    const Index feat = 4;
    const Shape input_shape{seq, feat};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(input_shape, PoolingMethod::AveragePooling, "avg_pool"));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}},
                          {{8, 7, 6, 5}, {4, 3, 2, 1}, {0, 0, 0, 0}}});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], feat);

    const float expected[8] = {8.0f / 3.0f, 10.0f / 3.0f, 4.0f, 14.0f / 3.0f,
                               6.0f, 5.0f, 4.0f, 3.0f};
    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], expected[i], 1e-5f);
}

TEST(Pool3dOperatoreratorTest, ForwardAverageAllPaddingIsZero)
{
    const Index batch_size = 1;
    const Index seq = 2;
    const Index feat = 3;
    const Shape input_shape{seq, feat};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(input_shape, PoolingMethod::AveragePooling, "avg_pool"));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setZero();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], 0.0f, 1e-7f);
}

TEST(Pool3dOperatoreratorTest, BackPropagateMaxGradient)
{
    const Index samples_number = 6;
    const Index seq = 4;
    const Index feat = 3;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, Shape{seq, feat}, Shape{targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(Shape{seq, feat}, PoolingMethod::MaxPooling, "max_pool"));
    const Shape pool_output_shape = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(pool_output_shape, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(Pool3dOperatoreratorTest, BackPropagateAverageGradient)
{
    const Index samples_number = 6;
    const Index seq = 4;
    const Index feat = 3;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, Shape{seq, feat}, Shape{targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(Shape{seq, feat}, PoolingMethod::AveragePooling, "avg_pool"));
    const Shape pool_output_shape = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(pool_output_shape, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    const type max_abs_diff = (gradient - numerical_gradient).array().abs().maxCoeff();
    const type gradient_scale = max(type(1), numerical_gradient.array().abs().maxCoeff());
    EXPECT_LT(max_abs_diff / gradient_scale, type(2.0e-2));
}
