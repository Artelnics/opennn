#include "pch.h"
#include "numerical_derivatives.h"

#include <cmath>

#include "opennn/tensor_types.h"
#include "opennn/random_utilities.h"
#include "opennn/activation_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

static float silu(float x) { return x / (1.0f + std::exp(-x)); }


TEST(SiLUTest, ActivationForwardMatchesSilu)
{
    const Index features = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Activation>(Shape{features}, "SiLU", "act"));
    neural_network.compile();

    Tensor2 inputs(1, features);
    inputs.data()[0] = type(-1);
    inputs.data()[1] = type(0);
    inputs.data()[2] = type(1);
    inputs.data()[3] = type(2);

    ForwardPropagation forward_propagation(1, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs.data(), {1, features}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const float* output = forward_propagation.get_outputs().as<type>();
    EXPECT_NEAR(output[0], silu(-1.0f), 1.0e-5f);
    EXPECT_NEAR(output[1], silu(0.0f),  1.0e-5f);
    EXPECT_NEAR(output[2], silu(1.0f),  1.0e-5f);
    EXPECT_NEAR(output[3], silu(2.0f),  1.0e-5f);
}


TEST(SiLUTest, ActivationGradientMatchesNumerical)
{
    const Index samples_number = 6;
    const Index features = 5;
    const Index targets_number = 2;

    const Index hidden = 4;

    TabularDataset dataset(samples_number, Shape{features}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    // SiLU fused inside Dense (the realistic path; exercises the needs_input backward).
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{features}, Shape{hidden}, "SiLU"), {-1});
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{hidden}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}


// Parameter views: gate weight first, up weight second (the .bin layout of a
// separate gate/up Dense pair).
TEST(GatedDenseTest, ForwardMatchesHandComputed)
{
    const Index batch_size = 1;
    const Index features = 2;
    const Index outputs = 3;

    auto dense = make_unique<opennn::Dense>(Shape{features}, Shape{outputs}, "Identity", false, "gate_up");
    dense->set_use_bias(false);
    dense->set_gated(true);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(dense));
    neural_network.compile();

    // Weights are [in, out] row-major: w[i * outputs + j].
    const vector<float> gate_weights = { 0.5f, -1.0f,  2.0f,
                                         1.0f,  0.25f, -0.5f };
    const vector<float> up_weights   = { 1.5f,  2.0f, -1.0f,
                                        -0.5f,  1.0f,  0.75f };

    auto& views = neural_network.get_layer(Index(0))->get_parameter_views();
    ASSERT_EQ(views.size(), size_t(2));
    copy(gate_weights.begin(), gate_weights.end(), views[0].as<float>());
    copy(up_weights.begin(),   up_weights.end(),   views[1].as<float>());

    const vector<float> x = { 1.0f, -2.0f };

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<float> input_values = x;
    vector<TensorView> input_views = { TensorView(input_values.data(), {batch_size, features}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const float* output = forward_propagation.get_outputs().as<type>();

    for (Index j = 0; j < outputs; ++j)
    {
        float gate = 0.0f, up = 0.0f;
        for (Index i = 0; i < features; ++i)
        {
            gate += x[size_t(i)] * gate_weights[size_t(i * outputs + j)];
            up   += x[size_t(i)] * up_weights[size_t(i * outputs + j)];
        }
        EXPECT_NEAR(output[j], silu(gate) * up, 1.0e-5f);
    }
}


namespace
{

// Exercises both weight gradients and the accumulated input delta of the two
// projections sharing one input.
float gated_dense_max_gradient_error(bool use_bias)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index hidden = 4;
    const Index intermediate = 6;
    const Index targets_number = 2;

    const Shape input_shape{sequence_length, hidden};

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    auto gate_up = make_unique<opennn::Dense>(input_shape, Shape{intermediate}, "Identity", false, "gate_up");
    gate_up->set_use_bias(use_bias);
    gate_up->set_gated(true);
    neural_network.add_layer(move(gate_up), {-1});
    const Index gated_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(gated_index)->get_output_shape()), {gated_index});
    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    return (gradient - numerical_gradient).array().abs().maxCoeff();
}

}


TEST(GatedDenseTest, GradientMatchesNumerical)
{
    EXPECT_LT(gated_dense_max_gradient_error(/*use_bias*/ false), type(2.0e-3));
}


TEST(GatedDenseTest, GradientMatchesNumericalWithBias)
{
    EXPECT_LT(gated_dense_max_gradient_error(/*use_bias*/ true), type(2.0e-3));
}


TEST(GatedDenseTest, SaveLoadRoundTrip)
{
    const Index batch_size = 2;
    const Index sequence_length = 3;
    const Index hidden = 4;
    const Index intermediate = 5;

    auto gate_up = make_unique<opennn::Dense>(Shape{sequence_length, hidden}, Shape{intermediate},
                                              "Identity", false, "gate_up");
    gate_up->set_use_bias(false);
    gate_up->set_gated(true);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(gate_up));
    neural_network.compile();
    neural_network.set_parameters_random();

    vector<float> input_values(size_t(batch_size * sequence_length * hidden));
    for (size_t i = 0; i < input_values.size(); ++i)
        input_values[i] = 0.2f * float(i) - 1.0f;

    auto make_inputs = [&]() {
        return vector<TensorView>{ TensorView(input_values.data(), {batch_size, sequence_length, hidden}) };
    };

    ForwardPropagation forward_before(batch_size, &neural_network);
    vector<TensorView> inputs_before = make_inputs();
    neural_network.forward_propagate(inputs_before, forward_before, false);
    const TensorView out_before = forward_before.get_outputs();
    const vector<float> expected(out_before.as<float>(), out_before.as<float>() + out_before.size());

    const string path = (filesystem::temp_directory_path() / "opennn_gated_dense_roundtrip.json").string();
    neural_network.save(path);

    NeuralNetwork loaded;
    loaded.load(path);

    const auto* dense = dynamic_cast<const opennn::Dense*>(loaded.get_layer(Index(0)).get());
    ASSERT_NE(dense, nullptr);
    EXPECT_TRUE(dense->get_gated());
    EXPECT_FALSE(dense->get_use_bias());

    ForwardPropagation forward_after(batch_size, &loaded);
    vector<TensorView> inputs_after = make_inputs();
    loaded.forward_propagate(inputs_after, forward_after, false);
    const TensorView out_after = forward_after.get_outputs();

    ASSERT_EQ(out_after.size(), Index(expected.size()));
    for (Index i = 0; i < out_after.size(); ++i)
        EXPECT_NEAR(out_after.as<float>()[i], expected[size_t(i)], 1.0e-5f);

    error_code file_error;
    filesystem::remove(path, file_error);
    filesystem::remove(filesystem::path(path).replace_extension(".bin"), file_error);
}
