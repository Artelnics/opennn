#include "pch.h"
#include "numerical_derivatives.h"

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/kernel.cuh"
#include "opennn/core/random_utilities.h"

using namespace opennn;

namespace
{

VectorR read_host_parameters(const NeuralNetwork& network)
{
    return network.get_parameters_map();
}

float relative_difference(const VectorR& reference, const VectorR& other)
{
    const float max_abs_diff = (reference - other).array().abs().maxCoeff();
    const float scale = max(1.0f, reference.array().abs().maxCoeff());
    return max_abs_diff / scale;
}

float relative_difference(const MatrixR& reference, const MatrixR& other)
{
    const float max_abs_diff = (reference - other).array().abs().maxCoeff();
    const float scale = max(1.0f, reference.array().abs().maxCoeff());
    return max_abs_diff / scale;
}

}

class GpuComparison : public ::testing::Test
{
};

TEST_F(GpuComparison, ApproximationForward)
{
    const Index samples_number = 5;
    const Index inputs_number = 4;
    const Index outputs_number = 3;

    MatrixR inputs(samples_number, inputs_number);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    ApproximationNetwork cpu_network({inputs_number}, {6, 5}, {outputs_number});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ApproximationNetwork gpu_network({inputs_number}, {6, 5}, {outputs_number});
    gpu_network.set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network.calculate_outputs(inputs);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    EXPECT_LT(relative_difference(cpu_outputs, gpu_outputs), 1.0e-3f);
}

TEST_F(GpuComparison, ApproximationGradient)
{
    const Index samples_number = 6;
    const Index inputs_number = 4;
    const Index outputs_number = 2;

    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork cpu_network({inputs_number}, {6, 5}, {outputs_number});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ApproximationNetwork gpu_network({inputs_number}, {6, 5}, {outputs_number});
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, DenseGeluTanhFusedForward)
{
    const Index samples_number = 5;
    const Index inputs_number = 4;
    const Index hidden_number = 16;
    const Index outputs_number = 3;

    MatrixR inputs(samples_number, inputs_number);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    NeuralNetwork cpu_network;
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELUTanh"));
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    cpu_network.compile();
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELUTanh"));
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    gpu_network.compile();
    gpu_network.set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network.calculate_outputs(inputs);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    EXPECT_LT(relative_difference(cpu_outputs, gpu_outputs), 1.0e-3f);
}

TEST_F(GpuComparison, DenseGeluTanhFusedGradient)
{
    const Index samples_number = 6;
    const Index inputs_number = 4;
    const Index hidden_number = 16;
    const Index outputs_number = 2;

    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork cpu_network;
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELUTanh"));
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    cpu_network.compile();
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELUTanh"));
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    gpu_network.compile();
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

namespace
{
struct ScopedDreluFusionEnv
{
    ScopedDreluFusionEnv()  { setenv("OPENNN_DRELU_FUSION", "1", 1); }
    ~ScopedDreluFusionEnv() { unsetenv("OPENNN_DRELU_FUSION"); }
};
}

TEST_F(GpuComparison, DenseDreluFusedGradient)
{
    const ScopedDreluFusionEnv fusion_env;

    const Index samples_number = 6;
    const Index inputs_number = 4;
    const Index hidden_number = 128;
    const Index outputs_number = 2;

    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork cpu_network;
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "ReLU"));
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{hidden_number}, "ReLU"));
    cpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    cpu_network.compile();
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "ReLU"));
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{hidden_number}, "ReLU"));
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    gpu_network.compile();
    gpu_network.set_parameters(parameters);

    const auto* hidden_2 = dynamic_cast<const opennn::Dense*>(gpu_network.get_layers()[1].get());
    const auto* output_layer = dynamic_cast<const opennn::Dense*>(gpu_network.get_layers()[2].get());
    ASSERT_NE(hidden_2, nullptr);
    ASSERT_NE(output_layer, nullptr);
    EXPECT_TRUE(hidden_2->drelu_fusion_wired());
    EXPECT_TRUE(output_layer->drelu_fusion_wired());

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    EXPECT_TRUE(hidden_2->drelu_fusion_ran());
    EXPECT_TRUE(output_layer->drelu_fusion_ran());

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, ClassificationForward)
{
    const Index samples_number = 5;
    const Index inputs_number = 4;
    const Index classes_number = 3;

    MatrixR inputs(samples_number, inputs_number);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    ClassificationNetwork cpu_network({inputs_number}, {6}, {classes_number});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ClassificationNetwork gpu_network({inputs_number}, {6}, {classes_number});
    gpu_network.set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network.calculate_outputs(inputs);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    EXPECT_LT(relative_difference(cpu_outputs, gpu_outputs), 1.0e-3f);
}

TEST_F(GpuComparison, ClassificationGradient)
{
    const Index samples_number = 6;
    const Index inputs_number = 4;
    const Index classes_number = 3;

    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(samples_number, {inputs_number}, {classes_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ClassificationNetwork cpu_network({inputs_number}, {6}, {classes_number});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ClassificationNetwork gpu_network({inputs_number}, {6}, {classes_number});
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, ImageClassificationForward)
{
    const Index samples_number = 3;
    const Index height = 12;
    const Index width = 12;
    const Index channels = 3;
    const Index classes_number = 4;

    Tensor4 inputs(samples_number, height, width, channels);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    ImageClassificationNetwork cpu_network({height, width, channels}, {4, 8}, {classes_number});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ImageClassificationNetwork gpu_network({height, width, channels}, {4, 8}, {classes_number});
    gpu_network.set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network.calculate_outputs(inputs);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    EXPECT_LT(relative_difference(cpu_outputs, gpu_outputs), 1.0e-3f);
}

TEST_F(GpuComparison, ImageClassificationGradient)
{
    const Index samples_number = 4;
    const Index height = 12;
    const Index width = 12;
    const Index channels = 3;
    const Index classes_number = 4;

    Configuration::instance().set(Device::CPU, Type::FP32);
    TabularDataset dataset(samples_number, {height, width, channels}, {classes_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ImageClassificationNetwork cpu_network({height, width, channels}, {4, 8}, {classes_number});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ImageClassificationNetwork gpu_network({height, width, channels}, {4, 8}, {classes_number});
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 5.0e-3f);
}

TEST_F(GpuComparison, ProjectionResidualGradient)
{
    constexpr Index samples_number = 4;
    const Shape input_shape{2, 2, 8};

    TabularDataset dataset(samples_number, input_shape, Shape{1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    const auto build_network = [&](NeuralNetwork& network)
    {
        network.add_layer(make_unique<Convolutional>(
                              input_shape, Shape{1, 1, 8, 64}, "ReLU",
                              Shape{1, 1}, "Same", true, "stem"),
                          {-1});
        network.add_layer(make_unique<Convolutional>(
                              Shape{2, 2, 64}, Shape{1, 1, 64, 64}, "ReLU",
                              Shape{1, 1}, "Same", true, "main"),
                          {0});
        network.add_layer(make_unique<Convolutional>(
                              Shape{2, 2, 64}, Shape{1, 1, 64, 64}, "Identity",
                              Shape{1, 1}, "Same", true, "projection"),
                          {0});

        auto residual = make_unique<Convolutional>(
            Shape{2, 2, 64}, Shape{1, 1, 64, 64}, "ReLU",
            Shape{1, 1}, "Same", true, "residual");
        residual->set_residual(true);
        network.add_layer(move(residual), {1, 2});

        network.add_layer(make_unique<Convolutional>(
                              Shape{2, 2, 64}, Shape{1, 1, 64, 8}, "ReLU",
                              Shape{1, 1}, "Same", true, "later"),
                          {3});
        network.add_layer(make_unique<Flatten>(Shape{2, 2, 8}), {4});
        network.add_layer(make_unique<opennn::Dense>(
                              Shape{32}, Shape{1}, "Identity"),
                          {5});
        network.compile();
        network.set_training_activation_recomputation(true);
    };

    Configuration::instance().set(Device::CPU, Type::FP32);
    NeuralNetwork cpu_network;
    build_network(cpu_network);
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    build_network(gpu_network);
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 5.0e-3f);
}

TEST_F(GpuComparison, ResidentInferenceGraphReplay)
{
    const Index samples_number = 4;
    const Index height = 32;
    const Index width = 32;
    const Index channels = 3;
    const Index classes_number = 5;

    Tensor4 inputs(samples_number, height, width, channels);
    inputs.setRandom();

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ResNet network({height, width, channels}, {1, 1, 1, 1}, Shape{8, 16, 32, 64},
                   Shape{classes_number}, true);
    network.set_parameters_random();

    const Index input_bytes = inputs.size() * Index(sizeof(float));
    Buffer input_buffer;
    input_buffer.resize_bytes(input_bytes, Device::CUDA);
    device::copy_async(input_buffer.data, inputs.data(), input_bytes,
                       device::CopyKind::HostToDevice);
    device::synchronize();

    const TensorView input_view(input_buffer.data,
                                Shape{samples_number, height, width, channels},
                                Type::FP32, Device::CUDA);

    ForwardPropagation forward_propagation(samples_number, &network);

    const auto read_outputs = [](const TensorView& outputs)
    {
        vector<float> host(size_t(outputs.size()));
        device::synchronize();
        copy_device_to_host_float(outputs.data, outputs.type, outputs.size(),
                                  host.data(), Backend::get_compute_stream());
        device::synchronize();
        return host;
    };

    network.calculate_outputs_resident({input_view}, forward_propagation, true);
    const TensorView eager_view =
        network.calculate_outputs_resident({input_view}, forward_propagation, false);
    const vector<float> reference = read_outputs(eager_view);

    forward_propagation.set_cuda_graph(true);
    network.calculate_outputs_resident({input_view}, forward_propagation, true);
    network.calculate_outputs_resident({input_view}, forward_propagation, false);
    ASSERT_TRUE(static_cast<bool>(forward_propagation.inference_graph_exec));

    for (Index i = 0; i < 3; ++i)
    {
        const TensorView replay_view =
            network.calculate_outputs_resident({input_view}, forward_propagation, false);
        const vector<float> replayed = read_outputs(replay_view);

        ASSERT_EQ(reference.size(), replayed.size());
        for (size_t j = 0; j < reference.size(); ++j)
            ASSERT_NEAR(reference[j], replayed[j], 1.0e-6f);
    }

}

TEST_F(GpuComparison, ResidentInferenceGraphInvalidation)
{
    const Index samples_number = 4;
    const Index height = 32;
    const Index width = 32;
    const Index channels = 3;
    const Index classes_number = 5;

    Tensor4 inputs(samples_number, height, width, channels);
    inputs.setRandom();

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ResNet network({height, width, channels}, {1, 1, 1, 1}, Shape{8, 16, 32, 64},
                   Shape{classes_number}, true);
    network.set_parameters_random();
    VectorR shifted_parameters = read_host_parameters(network);
    shifted_parameters.array() += 0.25f;

    const Index input_bytes = inputs.size() * Index(sizeof(float));
    Buffer input_buffer;
    input_buffer.resize_bytes(input_bytes, Device::CUDA);
    device::copy_async(input_buffer.data, inputs.data(), input_bytes,
                       device::CopyKind::HostToDevice);
    Buffer second_input_buffer;
    second_input_buffer.resize_bytes(input_bytes, Device::CUDA);
    device::copy_async(second_input_buffer.data, inputs.data(), input_bytes,
                       device::CopyKind::HostToDevice);
    device::synchronize();

    const TensorView input_view(input_buffer.data,
                                Shape{samples_number, height, width, channels},
                                Type::FP32, Device::CUDA);
    const TensorView second_input_view(second_input_buffer.data,
                                       Shape{samples_number, height, width, channels},
                                       Type::FP32, Device::CUDA);

    ForwardPropagation forward_propagation(samples_number, &network);

    const auto read_outputs = [](const TensorView& outputs)
    {
        vector<float> host(size_t(outputs.size()));
        device::synchronize();
        copy_device_to_host_float(outputs.data, outputs.type, outputs.size(),
                                  host.data(), Backend::get_compute_stream());
        device::synchronize();
        return host;
    };

    forward_propagation.set_cuda_graph(true);
    network.calculate_outputs_resident({input_view}, forward_propagation, true);
    const TensorView first_eager_view =
        network.calculate_outputs_resident({input_view}, forward_propagation, false);
    const vector<float> first_outputs = read_outputs(first_eager_view);
    ASSERT_TRUE(static_cast<bool>(forward_propagation.inference_graph_exec));

    const TensorView mismatch_view =
        network.calculate_outputs_resident({second_input_view}, forward_propagation, false);
    const vector<float> mismatch_outputs = read_outputs(mismatch_view);
    ASSERT_TRUE(static_cast<bool>(forward_propagation.inference_graph_exec));
    for (size_t j = 0; j < first_outputs.size(); ++j)
        ASSERT_NEAR(first_outputs[j], mismatch_outputs[j], 1.0e-6f);

    network.set_parameters(shifted_parameters);
    network.calculate_outputs_resident({input_view}, forward_propagation, true);
    ASSERT_FALSE(static_cast<bool>(forward_propagation.inference_graph_exec));
    const TensorView second_eager_view =
        network.calculate_outputs_resident({input_view}, forward_propagation, false);
    const vector<float> second_reference = read_outputs(second_eager_view);
    ASSERT_TRUE(static_cast<bool>(forward_propagation.inference_graph_exec));

    const TensorView replay_view =
        network.calculate_outputs_resident({input_view}, forward_propagation, false);
    const vector<float> replayed = read_outputs(replay_view);

    float max_change = 0.0f;
    for (size_t j = 0; j < first_outputs.size(); ++j)
        max_change = max(max_change, abs(first_outputs[j] - second_reference[j]));
    EXPECT_GT(max_change, 1.0e-4f);

    for (size_t j = 0; j < second_reference.size(); ++j)
        ASSERT_NEAR(second_reference[j], replayed[j], 1.0e-6f);

}

TEST_F(GpuComparison, ForecastingRecurrentForward)
{
    const Index samples_number = 7;
    const Index past = 5;
    const Index features = 3;

    Tensor3 inputs(samples_number, past, features);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    ForecastingNetwork cpu_network({past, features}, {6, 5}, {1});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingNetwork gpu_network({past, features}, {6, 5}, {1});
    gpu_network.set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network.calculate_outputs(inputs);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    EXPECT_LT(relative_difference(cpu_outputs, gpu_outputs), 1.0e-3f);
}

TEST_F(GpuComparison, ForecastingLstmForward)
{
    const Index samples_number = 7;
    const Index past = 5;
    const Index features = 3;

    Tensor3 inputs(samples_number, past, features);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    ForecastingLstmNetwork cpu_network({past, features}, {6, 5}, {1});
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingLstmNetwork gpu_network({past, features}, {6, 5}, {1});
    gpu_network.set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network.calculate_outputs(inputs);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    EXPECT_LT(relative_difference(cpu_outputs, gpu_outputs), 1.0e-3f);
}

TEST_F(GpuComparison, ForecastingRecurrentGradient)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(7);
    TimeSeriesDataset dataset(30, {2}, {1});
    dataset.set_data_random();
    dataset.set_past_time_steps(5);
    dataset.set_future_time_steps(1);
    dataset.set_sample_roles("Training");

    ForecastingNetwork cpu_network(dataset.get_input_shape(), {6, 5}, dataset.get_target_shape());
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingNetwork gpu_network(dataset.get_input_shape(), {6, 5}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, ForecastingLstmGradient)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(7);
    TimeSeriesDataset dataset(30, {2}, {1});
    dataset.set_data_random();
    dataset.set_past_time_steps(5);
    dataset.set_future_time_steps(1);
    dataset.set_sample_roles("Training");

    ForecastingLstmNetwork cpu_network(dataset.get_input_shape(), {6, 5}, dataset.get_target_shape());
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingLstmNetwork gpu_network(dataset.get_input_shape(), {6, 5}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, ForecastingLstmFusedGradient)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(11);
    TimeSeriesDataset dataset(40, {2}, {1});
    dataset.set_data_random();
    dataset.set_past_time_steps(6);
    dataset.set_future_time_steps(1);
    dataset.set_sample_roles("Training");

    ForecastingLstmNetwork cpu_network(dataset.get_input_shape(), {64}, dataset.get_target_shape());
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingLstmNetwork gpu_network(dataset.get_input_shape(), {64}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, ForecastingRecurrentWideGradient)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(11);
    TimeSeriesDataset dataset(40, {2}, {1});
    dataset.set_data_random();
    dataset.set_past_time_steps(6);
    dataset.set_future_time_steps(1);
    dataset.set_sample_roles("Training");

    ForecastingNetwork cpu_network(dataset.get_input_shape(), {64}, dataset.get_target_shape());
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingNetwork gpu_network(dataset.get_input_shape(), {64}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);
}

TEST_F(GpuComparison, TransformerForward)
{
    const Index batch_size = 2;
    const Index input_sequence_length = 4;
    const Index decoder_sequence_length = 3;
    const Index input_vocabulary_size = 12;
    const Index output_vocabulary_size = 14;
    const Index embedding_dimension = 8;
    const Index heads_number = 2;
    const Index feed_forward_dimension = 16;
    const Index layers_number = 1;

    Tensor3 decoder_inputs(batch_size, decoder_sequence_length, 1);
    Tensor3 encoder_inputs(batch_size, input_sequence_length, 1);

    for (Index i = 0; i < batch_size; ++i)
    {
        for (Index j = 0; j < decoder_sequence_length; ++j)
            decoder_inputs(i, j, 0) = float(1 + (i + j) % (output_vocabulary_size - 1));
        for (Index j = 0; j < input_sequence_length; ++j)
            encoder_inputs(i, j, 0) = float(1 + (i * 2 + j) % (input_vocabulary_size - 1));
    }

    Configuration::instance().set(Device::CPU, Type::FP32);
    Transformer cpu_network(input_sequence_length, decoder_sequence_length,
                            input_vocabulary_size, output_vocabulary_size,
                            embedding_dimension, heads_number,
                            feed_forward_dimension, layers_number);
    cpu_network.set_dropout_rate(0.0f);
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const Tensor3 cpu_outputs = cpu_network.calculate_outputs(decoder_inputs, encoder_inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    Transformer gpu_network(input_sequence_length, decoder_sequence_length,
                            input_vocabulary_size, output_vocabulary_size,
                            embedding_dimension, heads_number,
                            feed_forward_dimension, layers_number);
    gpu_network.set_dropout_rate(0.0f);
    gpu_network.set_parameters(parameters);
    const Tensor3 gpu_outputs = gpu_network.calculate_outputs(decoder_inputs, encoder_inputs);

    ASSERT_EQ(cpu_outputs.size(), gpu_outputs.size());
    const VectorR cpu_flat = Map<const VectorR>(cpu_outputs.data(), cpu_outputs.size());
    const VectorR gpu_flat = Map<const VectorR>(gpu_outputs.data(), gpu_outputs.size());
    EXPECT_LT(relative_difference(cpu_flat, gpu_flat), 1.0e-3f);
}

TEST_F(GpuComparison, SdpaAttentionRefreshesPaddingBetweenBatches)
{
    if (!AttentionOperator::sdpa_supported(Type::FP32, Device::CUDA))
        GTEST_SKIP() << "SDPA is not available in this build.";

    Configuration::instance().set(Device::CUDA, Type::FP32);
    set_seed(7);

    const Index batch_size = 2;
    const Index sequence_length = 64;
    const Index embedding_dimension = 64;
    const Index heads_number = 4;

    auto attention = make_unique<MultiHeadAttention>(
        Shape{sequence_length, embedding_dimension}, heads_number);
    attention->set_sdpa_min_sequence_length(1);

    NeuralNetwork network;
    network.add_layer(move(attention));
    network.compile();
    network.set_parameters_random();

    ASSERT_TRUE(static_cast<MultiHeadAttention*>(network.get_layer(0).get())->should_use_sdpa());

    const auto make_batch = [&](const std::array<Index, 2>& valid_lengths)
    {
        VectorR data(batch_size * sequence_length * embedding_dimension);
        data.setRandom();
        for (Index sample = 0; sample < batch_size; ++sample)
            for (Index row = valid_lengths[size_t(sample)]; row < sequence_length; ++row)
                data.segment((sample * sequence_length + row) * embedding_dimension,
                             embedding_dimension).setZero();
        return data;
    };

    VectorR batch_short = make_batch({24, 40});
    VectorR batch_long  = make_batch({64, 56});

    const auto forward_outputs = [&](ForwardPropagation& forward_propagation, VectorR& batch)
    {
        vector<TensorView> inputs = {
            TensorView(batch.data(), {batch_size, sequence_length, embedding_dimension})};
        network.forward_propagate(inputs, forward_propagation, true);

        const TensorView outputs = forward_propagation.get_outputs();
        VectorR host(outputs.size());
        device::copy_async(host.data(), outputs.data,
                           outputs.size() * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost,
                           Backend::get_compute_stream());
        device::synchronize(Backend::get_compute_stream());
        return host;
    };

    ForwardPropagation reused_propagation(batch_size, &network);
    forward_outputs(reused_propagation, batch_short);
    const VectorR outputs_after_reuse = forward_outputs(reused_propagation, batch_long);

    ForwardPropagation fresh_propagation(batch_size, &network);
    const VectorR outputs_fresh = forward_outputs(fresh_propagation, batch_long);

    ASSERT_EQ(outputs_after_reuse.size(), outputs_fresh.size());
    EXPECT_LT(relative_difference(outputs_fresh, outputs_after_reuse), 1.0e-5f);
}

TEST_F(GpuComparison, SdpaAttentionBackwardGradient)
{
    if (!AttentionOperator::sdpa_supported(Type::FP32, Device::CUDA))
        GTEST_SKIP() << "SDPA is not available in this build.";

    const Index samples_number = 4;
    const Index sequence_length = 64;
    const Index heads_number = 2;
    const Index embedding_dimension = 32;

    const Shape input_shape{sequence_length, embedding_dimension};

    Configuration::instance().set(Device::CPU, Type::FP32);
    set_seed(11);

    TabularDataset dataset(samples_number, input_shape, {sequence_length * embedding_dimension});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork cpu_network;
    cpu_network.add_layer(make_unique<MultiHeadAttention>(input_shape, heads_number));
    cpu_network.add_layer(make_unique<Flatten>(cpu_network.get_output_shape()));
    cpu_network.compile();
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    auto gpu_attention = make_unique<MultiHeadAttention>(input_shape, heads_number);
    gpu_attention->set_sdpa_min_sequence_length(1);
    gpu_network.add_layer(move(gpu_attention));
    gpu_network.add_layer(make_unique<Flatten>(gpu_network.get_output_shape()));
    gpu_network.compile();
    gpu_network.set_parameters(parameters);

    ASSERT_TRUE(static_cast<MultiHeadAttention*>(gpu_network.get_layer(0).get())->should_use_sdpa());

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 2.0e-2f);
}

#endif
