#include "pch.h"

#ifdef OPENNN_HAS_CUDA

#include "opennn/configuration.h"
#include "opennn/tensor_types.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/time_series_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/standard_networks.h"
#include "opennn/loss.h"
#include "opennn/forward_propagation.h"
#include "opennn/back_propagation.h"
#include "opennn/batch.h"
#include "opennn/device_backend.h"
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

#endif
