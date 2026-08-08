#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/long_short_term_memory_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"
#include "opennn/batch.h"
#include "opennn/forward_propagation.h"
#include "opennn/back_propagation.h"

#include <chrono>
#include <cmath>
#include <cstdio>

using namespace opennn;

namespace {

vector<double> reference_constant_outputs(const Index features,
                                          const Index neurons,
                                          const Index time_steps,
                                          const double c,
                                          const bool identity_cell)
{
    vector<double> outputs(time_steps);
    double h = 0.0;
    double cell = 0.0;

    for (Index t = 0; t < time_steps; ++t)
    {
        const double z = c * (1.0 + double(features) + double(neurons) * h);
        const double s = 1.0 / (1.0 + std::exp(-z));
        const double g = identity_cell ? z : std::tanh(z);
        cell = s * cell + s * g;
        const double a = identity_cell ? cell : std::tanh(cell);
        h = s * a;
        outputs[t] = h;
    }

    return outputs;
}

void check_constant_forward(const Index neurons, const string& cell_activation)
{
    const Index samples_number = 2;
    const Index features       = 8;
    const Index time_steps     = 6;
    const type  c              = type(0.02);

    NeuralNetwork neural_network;
    auto layer = make_unique<LongShortTermMemory>(
        Shape{time_steps, features}, Shape{neurons});
    layer->set_activation_function(cell_activation);
    layer->set_recurrent_activation_function("Sigmoid");
    layer->set_return_sequences(true);
    neural_network.add_layer(move(layer));
    neural_network.compile();

    VectorMap(neural_network.get_parameters_data(),
              neural_network.get_parameters_buffer_size()).setConstant(c);

    Tensor3 inputs(samples_number, time_steps, features);
    inputs.setConstant(type(1));

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    vector<TensorView> input_views = {
        TensorView(inputs.data(), {samples_number, time_steps, features})};
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView outputs_view = forward_propagation.get_outputs();
    ASSERT_EQ(outputs_view.shape.size(), samples_number * time_steps * neurons);

    const vector<double> reference = reference_constant_outputs(
        features, neurons, time_steps, double(c), cell_activation == "Identity");

    const type* outputs = outputs_view.as<type>();
    double max_difference = 0.0;

    for (Index b = 0; b < samples_number; ++b)
        for (Index t = 0; t < time_steps; ++t)
            for (Index h = 0; h < neurons; ++h)
                max_difference = max(max_difference,
                    abs(double(outputs[(b * time_steps + t) * neurons + h])
                             - reference[t]));

    EXPECT_LT(max_difference, 1.0e-4)
        << "H=" << neurons << " cell activation=" << cell_activation;
}

void set_varied_parameters(NeuralNetwork& neural_network)
{
    float* parameters = neural_network.get_parameters_data();
    const Index parameters_number = neural_network.get_parameters_buffer_size();

    for (Index i = 0; i < parameters_number; ++i)
        parameters[i] = 0.05f * std::sin(0.7f * float(i) + 0.3f);
}

void check_gradient(const Index neurons, const bool return_sequences)
{
    const Index samples_number = 4;
    const Index inputs_number  = 3;
    const Index time_steps     = 3;

    Shape target_shape{neurons};
    if (return_sequences) target_shape = Shape{time_steps, neurons};

    TabularDataset dataset(samples_number, {time_steps, inputs_number}, target_shape);
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    auto layer = make_unique<LongShortTermMemory>(
        Shape{time_steps, inputs_number}, Shape{neurons});
    layer->set_return_sequences(return_sequences);
    neural_network.add_layer(move(layer));
    neural_network.compile();
    set_varied_parameters(neural_network);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    EXPECT_GE(calculate_numerical_error(loss), 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3))
        << "H=" << neurons << " return_sequences=" << return_sequences;
}

}

TEST(LstmFusedPath, ForwardMatchesAcrossBoundary)
{
    check_constant_forward(64, "Tanh");
    check_constant_forward(96, "Tanh");
    check_constant_forward(64, "Identity");

    check_gradient(64, false);
    check_gradient(64, true);
    check_gradient(96, false);
}

TEST(LstmFusedPath, ScalarAndFusedAgree)
{

    check_constant_forward(63, "Tanh");
    check_constant_forward(64, "Tanh");

    check_gradient(63, false);
    check_gradient(64, false);
}

TEST(LstmFusedPath, DISABLED_BenchmarkBoundary)
{
    const Index samples_number = 32;
    const Index time_steps     = 24;
    const Index features       = 8;
    const int   warmup         = 5;
    const int   iterations     = 50;

    printf("batch=%lld T=%lld F=%lld iterations=%d\n",
                (long long)samples_number, (long long)time_steps,
                (long long)features, iterations);
    printf("%6s  %-6s  %10s  %10s\n", "H", "path", "fwd_us", "bwd_us");

    for (const Index neurons : {8, 16, 32, 48, 64, 96, 128})
    {
        TabularDataset dataset(samples_number, {time_steps, features}, {neurons});
        dataset.set_data_random();
        dataset.set_sample_roles("Training");

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<LongShortTermMemory>(
            Shape{time_steps, features}, Shape{neurons}));
        neural_network.compile();
        neural_network.set_parameters_glorot();

        Loss loss(&neural_network, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);

        Batch batch(samples_number, &dataset, neural_network.get_config());
        batch.fill(dataset.get_sample_indices("Training"),
                   dataset.get_feature_indices("Input"),
                   dataset.get_feature_indices("Decoder"),
                   dataset.get_feature_indices("Target"));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        BackPropagation back_propagation(samples_number, &loss);

        for (int i = 0; i < warmup; ++i)
        {
            neural_network.forward_propagate(batch.get_inputs(), forward_propagation, true);
            loss.back_propagate(batch, forward_propagation, back_propagation);
        }

        const auto t0 = chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            neural_network.forward_propagate(batch.get_inputs(), forward_propagation, true);
        const auto t1 = chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            loss.back_propagate(batch, forward_propagation, back_propagation);
        const auto t2 = chrono::steady_clock::now();

        const double forward_us =
            chrono::duration<double, micro>(t1 - t0).count() / iterations;
        const double backward_us =
            chrono::duration<double, micro>(t2 - t1).count() / iterations;

        printf("%6lld  %-6s  %10.1f  %10.1f\n",
                    (long long)neurons, neurons < 64 ? "scalar" : "fused",
                    forward_us, backward_us);
    }

    fflush(stdout);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
