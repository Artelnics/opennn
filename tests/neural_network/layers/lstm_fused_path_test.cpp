#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <utility>

#include "opennn/core/tensor_types.h"
#include "opennn/core/profiler.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/dataset/batch.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>

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
    neural_network.add_layer(std::move(layer));
    neural_network.compile();

    neural_network.get_parameters_map().setConstant(c);

    Tensor3 inputs(samples_number, time_steps, features);
    inputs.setConstant(type(1));

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    vector<TensorView> input_views = {
        TensorView(inputs.data(), {samples_number, time_steps, features})};
    neural_network.forward_propagate(input_views, forward_propagation, ForwardPropagationMode::Inference);

    const TensorView outputs_view = forward_propagation.get_outputs();
    ASSERT_EQ(outputs_view.get_shape().size(), samples_number * time_steps * neurons);

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
    VectorMap parameters = neural_network.get_parameters_map();

    for (Index i = 0; i < parameters.size(); ++i)
        parameters(i) = 0.05f * sin(0.7f * float(i) + 0.3f);
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
    neural_network.add_layer(std::move(layer));
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

void set_environment_variable(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

class ScopedEnvironmentVariable
{
public:
    ScopedEnvironmentVariable(const char* new_name, const char* value)
        : name(new_name)
    {
        if (const char* existing = getenv(name.c_str()))
        {
            had_original = true;
            original = existing;
        }
        set_environment_variable(name.c_str(), value);
    }

    ~ScopedEnvironmentVariable()
    {
        set_environment_variable(name.c_str(),
                                 had_original ? original.c_str() : nullptr);
    }

private:
    string name;
    string original;
    bool had_original = false;
};

class ScopedProfiler
{
public:
    ScopedProfiler() : was_enabled(profiler::is_enabled())
    {
        profiler::set_enabled(true);
    }

    ~ScopedProfiler()
    {
        profiler::set_enabled(was_enabled);
    }

private:
    bool was_enabled;
};

void check_onednn_gradient_against_scalar(const bool return_sequences)
{
    const Index samples_number = 4;
    const Index inputs_number  = 3;
    const Index time_steps     = 3;
    const Index neurons        = 128;

    Shape target_shape{neurons};
    if (return_sequences) target_shape = Shape{time_steps, neurons};

    TabularDataset dataset(samples_number, {time_steps, inputs_number}, target_shape);
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    auto layer = make_unique<LongShortTermMemory>(
        Shape{time_steps, inputs_number}, Shape{neurons});
    layer->set_return_sequences(return_sequences);
    neural_network.add_layer(std::move(layer));
    neural_network.compile();
    set_varied_parameters(neural_network);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    ScopedProfiler profile;
    const long forward_calls =
        profiler::stats().call_count("rnn:onednn_forward");
    const long backward_calls =
        profiler::stats().call_count("rnn:onednn_backward");

    VectorR onednn_gradient;
    {
        const ScopedEnvironmentVariable enable_onednn(
            "OPENNN_NO_ONEDNN_LSTM", nullptr);
        onednn_gradient = calculate_gradient(loss);
    }

    EXPECT_GT(profiler::stats().call_count("rnn:onednn_forward"),
              forward_calls);
    EXPECT_GT(profiler::stats().call_count("rnn:onednn_backward"),
              backward_calls);

    VectorR scalar_gradient;
    {
        const ScopedEnvironmentVariable disable_onednn(
            "OPENNN_NO_ONEDNN_LSTM", "1");
        scalar_gradient = calculate_gradient(loss);
    }

    EXPECT_LT((onednn_gradient - scalar_gradient).array().abs().maxCoeff(),
              type(1.0e-3))
        << "return_sequences=" << return_sequences;
}

void check_onednn_inference_cache(const bool return_sequences)
{
    const Index samples_number = 3;
    const Index inputs_number  = 4;
    const Index time_steps     = 5;
    const Index neurons        = 128;

    NeuralNetwork neural_network;
    auto layer = make_unique<LongShortTermMemory>(
        Shape{time_steps, inputs_number}, Shape{neurons});
    layer->set_return_sequences(return_sequences);
    neural_network.add_layer(std::move(layer));
    neural_network.compile();
    set_varied_parameters(neural_network);

    Tensor3 inputs(samples_number, time_steps, inputs_number);
    for (Index i = 0; i < inputs.size(); ++i)
        inputs.data()[i] = 0.1f * sin(0.13f * float(i) - 0.2f);

    const vector<TensorView> input_views{
        TensorView(inputs.data(), {samples_number, time_steps, inputs_number})};
    ForwardPropagation first(samples_number, &neural_network,
                             ForwardPropagationMode::Inference);
    ForwardPropagation second(samples_number, &neural_network,
                              ForwardPropagationMode::Inference);

    const auto run = [&](ForwardPropagation& forward_propagation)
    {
        neural_network.forward_propagate(
            input_views, forward_propagation, ForwardPropagationMode::Inference);
        const TensorView output = forward_propagation.get_outputs();
        VectorR snapshot(output.size());
        memcpy(snapshot.data(), output.get_data(),
               size_t(output.byte_size()));
        return snapshot;
    };

    ScopedProfiler profile;
    const long forward_calls =
        profiler::stats().call_count("rnn:onednn_forward");
    const long pack_calls =
        profiler::stats().call_count("rnn:onednn_pack_weights");

    VectorR original;
    VectorR repeated;
    VectorR second_original;
    VectorR updated;
    VectorR second_updated;
    {
        const ScopedEnvironmentVariable enable_onednn(
            "OPENNN_NO_ONEDNN_LSTM", nullptr);

        original = run(first);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_forward")
                      - forward_calls, 1);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_pack_weights")
                      - pack_calls, 1);

        repeated = run(first);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_forward")
                      - forward_calls, 2);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_pack_weights")
                      - pack_calls, 1);

        second_original = run(second);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_forward")
                      - forward_calls, 3);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_pack_weights")
                      - pack_calls, 2);

        {
            VectorMap parameters = neural_network.get_parameters_map();
            // Leave the four gate biases unchanged so a stale matrix cache
            // cannot be masked by the bias vector, which is packed every call.
            const Index biases_number = 4 * neurons;
            parameters.tail(parameters.size() - biases_number) =
                parameters.tail(parameters.size() - biases_number).array()
                    * -0.5f + 0.003f;
        }

        updated = run(first);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_forward")
                      - forward_calls, 4);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_pack_weights")
                      - pack_calls, 3);

        second_updated = run(second);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_forward")
                      - forward_calls, 5);
        EXPECT_EQ(profiler::stats().call_count("rnn:onednn_pack_weights")
                      - pack_calls, 4);
    }

    EXPECT_LT((original - repeated).array().abs().maxCoeff(), 1.0e-7f);
    EXPECT_LT((original - second_original).array().abs().maxCoeff(), 1.0e-7f);

    EXPECT_GT((original - updated).array().abs().maxCoeff(), 1.0e-4f);
    EXPECT_LT((updated - second_updated).array().abs().maxCoeff(), 1.0e-7f);

    ForwardPropagation scalar(samples_number, &neural_network,
                              ForwardPropagationMode::Inference);
    VectorR scalar_updated;
    {
        const ScopedEnvironmentVariable disable_onednn(
            "OPENNN_NO_ONEDNN_LSTM", "1");
        scalar_updated = run(scalar);
    }

    EXPECT_LT((updated - scalar_updated).array().abs().maxCoeff(), 2.0e-4f)
        << "return_sequences=" << return_sequences;
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

    check_constant_forward(95, "Tanh");
    check_constant_forward(96, "Tanh");

    check_gradient(95, false);
    check_gradient(96, false);
}

TEST(LstmFusedPath, OneDnnAnalyticGradientMatchesScalar)
{
#ifndef OPENNN_TEST_HAS_ONEDNN
    GTEST_SKIP() << "oneDNN support is disabled in this build";
#else
    check_constant_forward(128, "Tanh");
    check_onednn_gradient_against_scalar(false);
    check_onednn_gradient_against_scalar(true);
#endif
}

TEST(LstmFusedPath, OneDnnInferenceCacheTracksParameters)
{
#ifndef OPENNN_TEST_HAS_ONEDNN
    GTEST_SKIP() << "oneDNN support is disabled in this build";
#else
    check_onednn_inference_cache(false);
    check_onednn_inference_cache(true);
#endif
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
        batch.fill(dataset.get_sample_indices("Training"), dataset.get_feature_selection());

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        BackPropagation back_propagation(samples_number, loss);

        for (int i = 0; i < warmup; ++i)
        {
            neural_network.forward_propagate(batch.get_inputs(), forward_propagation, ForwardPropagationMode::Training);
            loss.back_propagate(batch, forward_propagation, back_propagation);
        }

        const auto t0 = chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            neural_network.forward_propagate(batch.get_inputs(), forward_propagation, ForwardPropagationMode::Training);
        const auto t1 = chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            loss.back_propagate(batch, forward_propagation, back_propagation);
        const auto t2 = chrono::steady_clock::now();

        const double forward_us =
            chrono::duration<double, micro>(t1 - t0).count() / iterations;
        const double backward_us =
            chrono::duration<double, micro>(t2 - t1).count() / iterations;

        printf("%6lld  %-6s  %10.1f  %10.1f\n",
                    (long long)neurons, neurons < 96 ? "scalar" : "fused",
                    forward_us, backward_us);
    }

    fflush(stdout);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
