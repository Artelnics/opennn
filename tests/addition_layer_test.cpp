#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/addition_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;


TEST(AdditionLayerTest, DefaultConstructor)
{
    Addition addition_layer;

    EXPECT_EQ(addition_layer.get_name(), "Addition");
    EXPECT_EQ(addition_layer.get_inputs_number(), 2);
}


TEST(AdditionLayerTest, GeneralConstructor)
{
    const Shape input_shape{4, 3};

    Addition addition_layer(input_shape, "residual_add", 3);

    EXPECT_EQ(addition_layer.get_name(), "Addition");
    EXPECT_EQ(addition_layer.get_label(), "residual_add");
    EXPECT_EQ(addition_layer.get_inputs_number(), 3);
    EXPECT_EQ(addition_layer.get_input_shape(), input_shape);
    EXPECT_EQ(addition_layer.get_output_shape(), input_shape);
}


TEST(AdditionLayerTest, OutputShapeMatchesInputShapeRank3)
{
    const Shape input_shape{5, 5, 8};

    Addition addition_layer(input_shape);

    EXPECT_EQ(addition_layer.get_output_shape(), input_shape);
    EXPECT_EQ(addition_layer.get_output_shape().size(), 5 * 5 * 8);
}


TEST(AdditionLayerTest, BackwardSpecsCountMatchesInputsNumber)
{
    const Index batch_size = 7;
    const Index inputs_number = 3;

    Addition addition_layer(Shape{4, 2}, "", inputs_number);

    const vector<TensorSpec> backward_specs = addition_layer.get_backward_specs(batch_size);

    EXPECT_EQ(ssize(backward_specs), inputs_number);

    for (const TensorSpec& spec : backward_specs)
    {
        const Shape& shape = spec.shape;
        ASSERT_EQ(shape.rank, 3);
        EXPECT_EQ(shape[0], batch_size);
        EXPECT_EQ(shape[1], 4);
        EXPECT_EQ(shape[2], 2);
    }
}


TEST(AdditionLayerTest, ForwardPropagateSumsTwoInputs)
{
    const Index batch_size = 2;
    const Index rows = 3;
    const Index cols = 2;
    const Index sample_size = rows * cols;
    const Index total = batch_size * sample_size;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Addition>(Shape{rows, cols}, "add", 2),
                             {-1, -2});
    neural_network.compile();

    Tensor3 input_a(batch_size, rows, cols);
    Tensor3 input_b(batch_size, rows, cols);

    for (Index i = 0; i < total; ++i)
    {
        input_a.data()[i] = float(i);
        input_b.data()[i] = float(2 * i + 1);
    }

    ForwardPropagation forward_propagation(batch_size, &neural_network);

    vector<TensorView> input_views = {
        TensorView(input_a.data(), {batch_size, rows, cols}),
        TensorView(input_b.data(), {batch_size, rows, cols})
    };

    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], rows);
    EXPECT_EQ(output_view.shape[2], cols);
    EXPECT_EQ(output_view.size(), total);

    const float* output_data = output_view.as<type>();

    for (Index i = 0; i < total; ++i)
        EXPECT_NEAR(output_data[i], input_a.data()[i] + input_b.data()[i], 1.0e-5f);
}


TEST(AdditionLayerTest, ForwardPropagateSumsThreeInputs)
{
    const Index batch_size = 2;
    const Index rows = 2;
    const Index cols = 2;
    const Index sample_size = rows * cols;
    const Index total = batch_size * sample_size;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Addition>(Shape{rows, cols}, "add3", 3),
                             {-1, -2, -3});
    neural_network.compile();

    Tensor3 input_a(batch_size, rows, cols);
    Tensor3 input_b(batch_size, rows, cols);
    Tensor3 input_c(batch_size, rows, cols);
    input_a.setConstant(1.0f);
    input_b.setConstant(2.0f);
    input_c.setConstant(4.0f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);

    vector<TensorView> input_views = {
        TensorView(input_a.data(), {batch_size, rows, cols}),
        TensorView(input_b.data(), {batch_size, rows, cols}),
        TensorView(input_c.data(), {batch_size, rows, cols})
    };

    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.size(), total);

    const float* output_data = output_view.as<type>();

    for (Index i = 0; i < total; ++i)
        EXPECT_NEAR(output_data[i], 7.0f, 1.0e-5f);
}


TEST(AdditionLayerTest, ResidualBackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<opennn::Dense>(input_shape, Shape{embedding_dimension}, "Identity"),
                             {-1});
    const Index dense_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Addition>(input_shape, "residual_add", 2),
                             {dense_index, -1});
    const Index addition_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(addition_index)->get_output_shape()),
                             {addition_index});

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
