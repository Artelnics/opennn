#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/dense_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;


// A bias-free Dense exposes only the weight parameter: input*output, no +output.
TEST(DenseNoBiasTest, ParameterCountExcludesBias)
{
    NeuralNetwork with_bias;
    with_bias.add_layer(make_unique<opennn::Dense>(Shape{4}, Shape{3}, "Identity"));
    with_bias.compile();

    NeuralNetwork without_bias;
    auto dense = make_unique<opennn::Dense>(Shape{4}, Shape{3}, "Identity");
    dense->set_use_bias(false);
    without_bias.add_layer(move(dense));
    without_bias.compile();

    EXPECT_EQ(with_bias.get_parameters_number(), 4 * 3 + 3);
    EXPECT_EQ(without_bias.get_parameters_number(), 4 * 3);
}


// Forward of a bias-free Dense is exactly x * W (no bias term). With an
// all-ones weight, output[j] = sum_i(input_i) for every j (independent of the
// weight storage order); a stray bias would shift this away from sum(input).
TEST(DenseNoBiasTest, ForwardIsPureMatmul)
{
    NeuralNetwork neural_network;
    auto dense = make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Identity");
    dense->set_use_bias(false);
    neural_network.add_layer(move(dense));
    neural_network.compile();

    VectorR parameters = VectorR::Ones(neural_network.get_parameters_size());
    neural_network.set_parameters(parameters);

    Tensor2 inputs(1, 2);
    inputs.data()[0] = type(1);
    inputs.data()[1] = type(2);

    ForwardPropagation forward_propagation(1, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs.data(), {1, 2}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const float* output = forward_propagation.get_outputs().as<type>();
    EXPECT_NEAR(output[0], type(3), 1.0e-5f);   // 1 + 2, no bias
    EXPECT_NEAR(output[1], type(3), 1.0e-5f);
    EXPECT_NEAR(output[2], type(3), 1.0e-5f);
}


TEST(DenseNoBiasTest, GradientMatchesNumerical)
{
    const Index samples_number = 6;
    const Index features = 5;
    const Index hidden = 4;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, Shape{features}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    auto dense = make_unique<opennn::Dense>(Shape{features}, Shape{hidden}, "Identity");
    dense->set_use_bias(false);
    neural_network.add_layer(move(dense), {-1});
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}
