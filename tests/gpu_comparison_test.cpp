#include "pch.h"

#ifdef OPENNN_HAS_CUDA

#include "opennn/configuration.h"
#include "opennn/tensor_types.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/time_series_dataset.h"
#include "opennn/dense_layer.h"
#include "opennn/neural_network.h"
#include "opennn/standard_networks.h"
#include "opennn/loss.h"
#include "opennn/forward_propagation.h"
#include "opennn/back_propagation.h"
#include "opennn/batch.h"
#include "opennn/device_backend.h"
#include "opennn/kernel.cuh"
#include "opennn/random_utilities.h"

using namespace opennn;

namespace
{

VectorR read_host_parameters(const NeuralNetwork& network)
{
    const Index size = network.get_parameters_size();
    VectorR parameters(size);
    const float* data = network.get_parameters_data();
    for (Index i = 0; i < size; ++i)
        parameters(i) = data[i];
    return parameters;
}

VectorR compute_gradient(Loss& loss)
{
    NeuralNetwork* neural_network = loss.get_neural_network();
    Dataset* dataset = loss.get_dataset();

    const Index samples_number = dataset->get_samples_number("Training");

    Batch batch(samples_number, dataset, neural_network->get_config());
    batch.fill(dataset->get_sample_indices("Training"),
               dataset->get_feature_indices("Input"),
               dataset->get_feature_indices("Decoder"),
               dataset->get_feature_indices("Target"));

    if (neural_network->is_gpu())
    {
        batch.copy_device_async(Backend::get_transfer_stream());
        batch.wait_h2d_complete();
    }

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation back_propagation(samples_number, &loss);

    neural_network->forward_propagate(batch.get_inputs(), forward_propagation, true);
    loss.back_propagate(batch, forward_propagation, back_propagation);

    back_propagation.gradient.migrate_to(Device::CPU);

    return Map<const VectorR, AlignedMax>(back_propagation.gradient.as<float>(),
                                          back_propagation.gradient.size_in_floats());
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
protected:

    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
    }
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

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ApproximationNetwork gpu_network({inputs_number}, {6, 5}, {outputs_number});
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "GELUTanh"));
    gpu_network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{outputs_number}, "Identity"));
    gpu_network.compile();
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ClassificationNetwork gpu_network({inputs_number}, {6}, {classes_number});
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ImageClassificationNetwork gpu_network({height, width, channels}, {4, 8}, {classes_number});
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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

    Configuration::instance().set(Device::CPU, Type::FP32);
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

    Configuration::instance().set(Device::CPU, Type::FP32);
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

    Configuration::instance().set(Device::CPU, Type::FP32);

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

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingNetwork gpu_network(dataset.get_input_shape(), {6, 5}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingLstmNetwork gpu_network(dataset.get_input_shape(), {6, 5}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingLstmNetwork gpu_network(dataset.get_input_shape(), {64}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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
    const VectorR cpu_gradient = compute_gradient(cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    ForecastingNetwork gpu_network(dataset.get_input_shape(), {64}, dataset.get_target_shape());
    gpu_network.set_parameters(parameters);

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = compute_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

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

    Configuration::instance().set(Device::CPU, Type::FP32);

    ASSERT_EQ(cpu_outputs.size(), gpu_outputs.size());
    const VectorR cpu_flat = Map<const VectorR>(cpu_outputs.data(), cpu_outputs.size());
    const VectorR gpu_flat = Map<const VectorR>(gpu_outputs.data(), gpu_outputs.size());
    EXPECT_LT(relative_difference(cpu_flat, gpu_flat), 1.0e-3f);
}

#ifndef OPENNN_NO_VISION

// Raw-kernel parity for the YOLOv8 anchor-free loss: the CPU kernels
// (yolo_v8_error_kernel / yolo_v8_gradient_kernel) against the CUDA pair
// (yolo_v8_error_cuda / yolo_v8_gradient_cuda) on identical data. Error
// accumulation order differs on the GPU (atomicAdd), hence relative tolerance.
TEST_F(GpuComparison, YoloV8LossParity)
{
    constexpr Index batch = 2;
    constexpr Index grid = 4;
    constexpr Index ncls = 3;
    constexpr Index ch_out = 4 + ncls;
    constexpr Index ch_tgt = 5 + ncls;
    const Index n_cells = batch * grid * grid;
    const Index out_floats = n_cells * ch_out;
    const Index tgt_floats = n_cells * ch_tgt;

    Configuration::instance().set(Device::CUDA, Type::FP32);

    // Deterministic mix of positive (flag 1), background (flag 0) and
    // ignore (flag -1) cells; predicted boxes overlap the ground truth.
    vector<float> out(size_t(out_floats), 0.0f);
    vector<float> tgt(size_t(tgt_floats), 0.0f);
    for (Index i = 0; i < n_cells; ++i)
    {
        float* o = out.data() + i * ch_out;
        float* t = tgt.data() + i * ch_tgt;

        o[0] = 0.35f + 0.3f * float(i % 3) / 3.0f;
        o[1] = 0.30f + 0.4f * float(i % 5) / 5.0f;
        o[2] = 0.15f + 0.02f * float(i % 4);
        o[3] = 0.12f + 0.03f * float(i % 3);
        for (Index c = 0; c < ncls; ++c)
            o[4 + c] = 0.1f + 0.8f * float((i + c) % 7) / 7.0f;

        if (i % 3 == 0)
        {
            t[0] = 0.5f; t[1] = 0.5f;
            t[2] = 0.18f; t[3] = 0.14f;
            t[4] = 1.0f;
            t[5 + (i % ncls)] = 1.0f;
        }
        else if (i % 7 == 1)
            t[4] = -1.0f;
    }

    const Shape out_shape({batch, grid, grid, ch_out});
    const Shape tgt_shape({batch, grid, grid, ch_tgt});
    const TensorView out_view(out.data(), out_shape, Type::FP32);
    const TensorView tgt_view(tgt.data(), tgt_shape, Type::FP32);

    const YoloLambdas lam{5.0f, 0.5f, 1.0f, 2.0f, 0.0f};
    const float inv_batch = 1.0f / float(batch);

    const float cpu_error = yolo_v8_error_kernel(out_view, tgt_view, ncls, lam);

    vector<float> cpu_delta(size_t(out_floats), 0.0f);
    const TensorView cpu_delta_view(cpu_delta.data(), out_shape, Type::FP32);
    yolo_v8_gradient_kernel(out_view, tgt_view, cpu_delta_view, ncls, inv_batch, lam);

    const Index out_bytes = out_floats * Index(sizeof(float));
    Buffer out_dev, tgt_dev, err_dev, delta_dev;
    out_dev.resize_bytes(out_bytes, Device::CUDA);
    tgt_dev.resize_bytes(tgt_floats * Index(sizeof(float)), Device::CUDA);
    err_dev.resize_bytes(Index(sizeof(float)), Device::CUDA);
    delta_dev.resize_bytes(out_bytes, Device::CUDA);
    device::copy_async(out_dev.data, out.data(), out_bytes, device::CopyKind::HostToDevice);
    device::copy_async(tgt_dev.data, tgt.data(), tgt_floats * Index(sizeof(float)),
                       device::CopyKind::HostToDevice);
    err_dev.setZero();
    device::synchronize();

    yolo_v8_error_cuda(out_dev.as<float>(), tgt_dev.as<float>(), err_dev.as<float>(),
                       int(batch), int(grid), int(ncls), lam.giou, lam.cls, lam.focal_gamma);
    yolo_v8_gradient_cuda(out_dev.as<float>(), tgt_dev.as<float>(), delta_dev.as<float>(),
                          int(batch), int(grid), int(ncls), inv_batch,
                          lam.giou, lam.cls, lam.focal_gamma);
    device::synchronize();

    float gpu_error = 0.0f;
    vector<float> gpu_delta(size_t(out_floats), 0.0f);
    device::copy_async(&gpu_error, err_dev.data, Index(sizeof(float)),
                       device::CopyKind::DeviceToHost);
    device::copy_async(gpu_delta.data(), delta_dev.data, out_bytes,
                       device::CopyKind::DeviceToHost);
    device::synchronize();

    Configuration::instance().set(Device::CPU, Type::FP32);

    const float error_scale = max(1.0f, abs(cpu_error));
    EXPECT_LT(abs(cpu_error - gpu_error) / error_scale, 1.0e-3f);

    const VectorR cpu_delta_flat = Map<const VectorR>(cpu_delta.data(), out_floats);
    const VectorR gpu_delta_flat = Map<const VectorR>(gpu_delta.data(), out_floats);
    EXPECT_LT(relative_difference(cpu_delta_flat, gpu_delta_flat), 1.0e-3f);
}

#endif // OPENNN_NO_VISION

#endif
