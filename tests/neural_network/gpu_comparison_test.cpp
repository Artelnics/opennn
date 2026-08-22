#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <barrier>
#include <future>
#include <utility>

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/neural_network/layers/pooling_layer_3d.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/flash_attention.cuh"
#include "opennn/core/cuda/kernel_prelude.cuh"
#include "opennn/core/random_utilities.h"
#include "opennn/registry.h"

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

template<typename Tensor, typename BuildNetwork>
void expect_concurrent_gpu_outputs(const Tensor& first_inputs,
                                   const Tensor& second_inputs,
                                   BuildNetwork&& build_network,
                                   float tolerance = 1.0e-3f)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    NeuralNetwork cpu_network;
    build_network(cpu_network);
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);
    const MatrixR first_reference = cpu_network.calculate_outputs(first_inputs);
    const MatrixR second_reference = cpu_network.calculate_outputs(second_inputs);

    Configuration::instance().set(Device::CUDA, Type::FP32);
    NeuralNetwork gpu_network;
    build_network(gpu_network);
    if (parameters.size() > 0)
        gpu_network.set_parameters(parameters);
    gpu_network.copy_parameters_device();

    barrier start(2);
    const auto calculate = [&](const Tensor& inputs)
    {
        start.arrive_and_wait();
        return gpu_network.calculate_outputs(inputs);
    };

    future<MatrixR> first = async(launch::async, calculate, cref(first_inputs));
    future<MatrixR> second = async(launch::async, calculate, cref(second_inputs));

    const MatrixR first_outputs = first.get();
    const MatrixR second_outputs = second.get();

    EXPECT_LT(relative_difference(first_reference, first_outputs), tolerance);
    EXPECT_LT(relative_difference(second_reference, second_outputs), tolerance);
}

class ExactInputProbe final : public Layer
{
public:
    ExactInputProbe()
        : Layer(LayerType::Activation, false)
    {
        input_shape = {1};
    }

    bool accepts_input_rank(Index rank) const override { return rank == 1; }
    bool allows_bf16_input_cast(size_t) const noexcept override { return false; }
    Shape get_output_shape() const override { return input_shape; }
    vector<TensorSpec> get_forward_specs(Index) const override { return {}; }

    void forward_propagate(ForwardPropagation& propagation, size_t layer, bool) override
    {
        called = true;
        observed_input = propagation.inputs[layer].front();
    }

    bool called = false;
    TensorView observed_input;
};

}

class GpuComparison : public ::testing::Test
{
};

TEST_F(GpuComparison, LayerContractPreservesExactExternalInput)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    auto probe = make_unique<ExactInputProbe>();
    ExactInputProbe* const probe_ptr = probe.get();

    NeuralNetwork network;
    network.add_layer(std::move(probe));
    network.compile();

    Tensor2 input(1, 1);
    input(0, 0) = 257.0f; // BF16 rounds this identifier to 256.

    ForwardPropagation propagation(1, &network, ForwardPropagationMode::Inference);
    network.forward_propagate({TensorView(input.data(), {1, 1})}, propagation, false);

    ASSERT_TRUE(probe_ptr->called);
    ASSERT_EQ(probe_ptr->observed_input.get_type(), Type::FP32);
    ASSERT_EQ(probe_ptr->observed_input.get_device(), Device::CUDA);

    float observed = 0.0f;
    device::copy_async(&observed, probe_ptr->observed_input.get_data(), sizeof(float),
                       device::CopyKind::DeviceToHost);
    device::synchronize();
    EXPECT_EQ(observed, 257.0f);
}

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

    MatrixR fusion_inputs(samples_number, inputs_number);
    fusion_inputs.setRandom();
    const vector<TensorView> fusion_input_views = {
        TensorView(fusion_inputs.data(), {samples_number, inputs_number})
    };
    ForwardPropagation first_forward(samples_number, &gpu_network);
    ForwardPropagation second_forward(samples_number, &gpu_network);
    gpu_network.forward_propagate(fusion_input_views, first_forward, true);
    gpu_network.forward_propagate(fusion_input_views, second_forward, true);

    ASSERT_EQ(first_forward.drelu_fused_by_layer[0], 1);
    ASSERT_EQ(first_forward.drelu_fused_by_layer[1], 1);
    ASSERT_EQ(second_forward.drelu_fused_by_layer[0], 1);

    const auto first_mask = ranges::find_if(
        first_forward.slots[0],
        [](const TensorView& slot) { return slot.is_int8() && !slot.empty(); });
    const auto second_mask = ranges::find_if(
        second_forward.slots[0],
        [](const TensorView& slot) { return slot.is_int8() && !slot.empty(); });
    ASSERT_NE(first_mask, first_forward.slots[0].end());
    ASSERT_NE(second_mask, second_forward.slots[0].end());
    EXPECT_NE(first_mask->get_data(), second_mask->get_data());

    Loss gpu_loss(&gpu_network, &dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

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

TEST_F(GpuComparison, ConvolutionGraphCacheSupportsConcurrentFirstUse)
{
    const Shape input_shape{12, 12, 3};

    Tensor4 first_inputs(2, 12, 12, 3);
    Tensor4 second_inputs(5, 12, 12, 3);
    first_inputs.setRandom();
    second_inputs.setRandom();

    const auto build_network = [&](NeuralNetwork& network)
    {
        network.add_layer(make_unique<Convolutional>(
                              input_shape, Shape{3, 3, 3, 8}, "ReLU",
                              Shape{1, 1}, "Same", false, "conv"),
                          {-1});
        network.compile();
    };

    expect_concurrent_gpu_outputs(first_inputs, second_inputs, build_network);
}

TEST_F(GpuComparison, PoolingDescriptorSupportsConcurrentFirstUse)
{
    struct RestoreRung
    {
        ~RestoreRung() { device::set_rung(device::MaxPoolingRung::Auto); }
    } restore;
    device::set_rung(device::MaxPoolingRung::Cudnn);

    const Shape input_shape{12, 12, 3};
    Tensor4 first_inputs(2, 12, 12, 3);
    Tensor4 second_inputs(5, 12, 12, 3);
    first_inputs.setRandom();
    second_inputs.setRandom();

    const auto build_network = [&](NeuralNetwork& network)
    {
        network.add_layer(make_unique<Pooling>(
                              input_shape, Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                              "MaxPooling", "pool"),
                          {-1});
        network.compile();
    };

    expect_concurrent_gpu_outputs(first_inputs, second_inputs, build_network);
}

TEST_F(GpuComparison, ImageClassificationForwardUnderWorkspaceCap)
{
    // Autotune now runs *within* the workspace cap: over-budget plans are barred
    // and left unbuilt, and the tuner only measures the survivors. The access
    // violation this combination used to fault with (sm_120, ResNet-50 batch 512,
    // 512 MiB cap) was Graph::get_autotune_workspace_size() dereferencing the
    // barred nullptr slots; cudnn_frontend_utilities::autotune_workspace_bytes
    // sizes the tuning scratch through the null-safe per-index query instead.
    // A 16 MiB cap on these shapes does remove plans, so this asserts the tuned-
    // under-cap path runs and still agrees with the CPU reference.
    struct RestoreConvolutionSettings
    {
        bool autotune;
        ~RestoreConvolutionSettings()
        {
            device::set_conv_workspace_cap(-1);
            device::set_conv_autotune(autotune);
        }
    } restore{device::conv_autotune_enabled()};

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

    device::set_conv_workspace_cap(int64_t(16) * 1024 * 1024);
    device::set_conv_autotune(true);
    ASSERT_EQ(device::conv_workspace_limit_bytes(), int64_t(16) * 1024 * 1024);

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
        network.add_layer(std::move(residual), {1, 2});

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

// The bf16 batch-norm backward has three rungs (fused native, plain native,
// FP32-staged); which one a shape takes depends on the cuDNN build, and a
// residual block on this GPU takes the plain one - the very configuration an
// in-source note once measured as producing wrong gradients. This pins each
// rung in turn on a ResNet-style residual block and checks the whole-network
// gradient against the CPU fp32 reference, in fp32 and in bf16, so "wrong"
// versus "bf16-rounded" is a number, not an opinion.
// A ResNet-style residual block, batch-normalized throughout: stem, 3x3 main
// path, and a 1x1 residual convolution whose BN adds the stem output before
// its ReLU - the wiring the fused BN forward/backward paths are built for.
static void build_residual_block(NeuralNetwork& network, const Shape& input_shape)
{
    network.add_layer(make_unique<Convolutional>(
                          input_shape, Shape{1, 1, input_shape[2], 64}, "ReLU",
                          Shape{1, 1}, "Same", true, "stem"),
                      {-1});
    network.add_layer(make_unique<Convolutional>(
                          Shape{4, 4, 64}, Shape{3, 3, 64, 64}, "ReLU",
                          Shape{1, 1}, "Same", true, "main"),
                      {0});
    auto residual = make_unique<Convolutional>(
        Shape{4, 4, 64}, Shape{1, 1, 64, 64}, "ReLU",
        Shape{1, 1}, "Same", true, "residual");
    residual->set_residual(true);
    network.add_layer(std::move(residual), {1, 0});
    network.add_layer(make_unique<Flatten>(Shape{4, 4, 64}), {2});
    network.add_layer(make_unique<opennn::Dense>(
                          Shape{1024}, Shape{1}, "Identity"),
                      {3});
    network.compile();
}

TEST_F(GpuComparison, ResidualBlockGradientBf16PerBackwardRung)
{
    constexpr Index samples_number = 8;
    const Shape input_shape{4, 4, 16};

    TabularDataset dataset(samples_number, input_shape, Shape{1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    const auto build_network = [&](NeuralNetwork& network) { build_residual_block(network, input_shape); };

    Configuration::instance().set(Device::CPU, Type::FP32);
    NeuralNetwork cpu_network;
    build_network(cpu_network);
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    struct RestoreRung
    {
        ~RestoreRung() { device::set_rung(device::BatchNormBackwardRung::Auto); }
    } restore;

    const auto gpu_gradient_on = [&](Type type, device::BatchNormBackwardRung rung)
    {
        device::set_rung(rung);
        Configuration::instance().set(Device::CUDA, type);
        NeuralNetwork gpu_network;
        build_network(gpu_network);
        gpu_network.set_parameters(parameters);
        Loss gpu_loss(&gpu_network, &dataset);
        gpu_loss.set_error(Loss::Error::MeanSquaredError);
        return calculate_gradient(gpu_loss);
    };

    // fp32 on the GPU first: this is the engine-correctness bar, free of any
    // precision question, and every rung must clear it.
    const VectorR fp32_auto  = gpu_gradient_on(Type::FP32, device::BatchNormBackwardRung::Auto);
    const VectorR fp32_plain = gpu_gradient_on(Type::FP32, device::BatchNormBackwardRung::PlainNative);
    const VectorR fp32_own   = gpu_gradient_on(Type::FP32, device::BatchNormBackwardRung::OwnKernel);

    const VectorR staged = gpu_gradient_on(Type::BF16, device::BatchNormBackwardRung::StagedFp32);
    const VectorR plain  = gpu_gradient_on(Type::BF16, device::BatchNormBackwardRung::PlainNative);
    const VectorR own    = gpu_gradient_on(Type::BF16, device::BatchNormBackwardRung::OwnKernel);
    const VectorR autor  = gpu_gradient_on(Type::BF16, device::BatchNormBackwardRung::Auto);

    for (const VectorR* gradient : {&fp32_auto, &fp32_plain, &fp32_own, &staged, &plain, &own, &autor})
        ASSERT_EQ(cpu_gradient.size(), gradient->size());

    const float fp32_auto_error  = relative_difference(cpu_gradient, fp32_auto);
    const float fp32_plain_error = relative_difference(cpu_gradient, fp32_plain);
    const float fp32_own_error   = relative_difference(cpu_gradient, fp32_own);
    const float staged_error     = relative_difference(cpu_gradient, staged);
    const float plain_error      = relative_difference(cpu_gradient, plain);
    const float own_error        = relative_difference(cpu_gradient, own);
    const float auto_error       = relative_difference(cpu_gradient, autor);
    const float plain_vs_staged  = relative_difference(staged, plain);
    const float own_vs_staged    = relative_difference(staged, own);

    cout << "residual-block gradient vs fp32 reference - fp32 GPU: auto " << fp32_auto_error
         << ", plain " << fp32_plain_error << ", own " << fp32_own_error
         << "; bf16 GPU: staged " << staged_error << ", plain " << plain_error
         << ", own " << own_error << ", auto " << auto_error
         << "; bf16 plain vs staged " << plain_vs_staged
         << ", own vs staged " << own_vs_staged << "\n";

    // fp32: the engines must reproduce the reference (measured ~1e-3).
    EXPECT_LT(fp32_auto_error, 5.0e-3f);
    EXPECT_LT(fp32_plain_error, 5.0e-3f);
    EXPECT_LT(fp32_own_error, 5.0e-3f);

    // bf16: the rungs compute the same thing and must agree with each other
    // (measured 1.6e-8 - identical). Against the fp32 reference they carry the
    // network's bf16 rounding (activations, deltas and weights at 8 mantissa
    // bits through a 3x3 conv, batch norm and a residual add: measured ~9%),
    // which is precision, not a fault; a broken engine is far above the bound.
    EXPECT_LT(plain_vs_staged, 1.0e-2f);
    EXPECT_LT(own_vs_staged, 1.0e-2f);
    EXPECT_LT(staged_error, 2.0e-1f);
    EXPECT_LT(plain_error, 2.0e-1f);
    EXPECT_LT(own_error, 2.0e-1f);
    EXPECT_LT(auto_error, 2.0e-1f);
}

// The library's own batch-norm training forward (BatchNormForwardRung::
// OwnKernel: batch statistics, running-statistics update, scale/shift with the
// residual add and ReLU, and the packed ReLU mask) against cuDNN's fused graph
// on the residual block, in fp32 and bf16. Two things are compared per rung:
// the whole-network gradient with the backward pinned to the library kernel -
// which gates dY by the mask when the forward left one and by Y otherwise, so
// the two paths meet here - and the inference output after that one training
// forward, which is where the running statistics show.
TEST_F(GpuComparison, ResidualBlockBatchNormForwardRungParity)
{
    constexpr Index samples_number = 8;
    const Shape input_shape{4, 4, 16};

    TabularDataset dataset(samples_number, input_shape, Shape{1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    Tensor4 inputs(samples_number, input_shape[0], input_shape[1], input_shape[2]);
    inputs.setRandom();

    Configuration::instance().set(Device::CPU, Type::FP32);
    NeuralNetwork cpu_network;
    build_residual_block(cpu_network, input_shape);
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    struct RestoreRungs
    {
        ~RestoreRungs()
        {
            device::set_rung(device::BatchNormForwardRung::Auto);
            device::set_rung(device::BatchNormBackwardRung::Auto);
        }
    } restore;
    device::set_rung(device::BatchNormBackwardRung::OwnKernel);

    struct Run { VectorR gradient; MatrixR outputs; };
    const auto gpu_run = [&](Type type, device::BatchNormForwardRung rung)
    {
        device::set_rung(rung);
        Configuration::instance().set(Device::CUDA, type);
        NeuralNetwork gpu_network;
        build_residual_block(gpu_network, input_shape);
        gpu_network.set_parameters(parameters);
        Loss gpu_loss(&gpu_network, &dataset);
        gpu_loss.set_error(Loss::Error::MeanSquaredError);
        Run run;
        run.gradient = calculate_gradient(gpu_loss);
        run.outputs = gpu_network.calculate_outputs(inputs);
        return run;
    };

    const Run fp32_cudnn = gpu_run(Type::FP32, device::BatchNormForwardRung::CudnnGraph);
    const Run fp32_own   = gpu_run(Type::FP32, device::BatchNormForwardRung::OwnKernel);
    const Run bf16_cudnn = gpu_run(Type::BF16, device::BatchNormForwardRung::CudnnGraph);
    const Run bf16_own   = gpu_run(Type::BF16, device::BatchNormForwardRung::OwnKernel);

    for (const Run* run : {&fp32_cudnn, &fp32_own, &bf16_cudnn, &bf16_own})
    {
        ASSERT_EQ(cpu_gradient.size(), run->gradient.size());
        ASSERT_EQ(cpu_outputs.rows(), run->outputs.rows());
        ASSERT_EQ(cpu_outputs.cols(), run->outputs.cols());
    }

    const float fp32_cudnn_error   = relative_difference(cpu_gradient, fp32_cudnn.gradient);
    const float fp32_own_error     = relative_difference(cpu_gradient, fp32_own.gradient);
    const float fp32_own_vs_cudnn  = relative_difference(fp32_cudnn.gradient, fp32_own.gradient);
    const float bf16_own_vs_cudnn  = relative_difference(bf16_cudnn.gradient, bf16_own.gradient);
    const float fp32_cudnn_outputs = relative_difference(cpu_outputs, fp32_cudnn.outputs);
    const float fp32_own_outputs   = relative_difference(cpu_outputs, fp32_own.outputs);
    const float bf16_own_outputs   = relative_difference(bf16_cudnn.outputs, bf16_own.outputs);

    cout << "batch-norm forward rung parity - fp32 gradient vs reference: cuDNN " << fp32_cudnn_error
         << ", own " << fp32_own_error << ", own vs cuDNN " << fp32_own_vs_cudnn
         << "; bf16 gradient own vs cuDNN " << bf16_own_vs_cudnn
         << "; inference after one step, fp32 vs reference: cuDNN " << fp32_cudnn_outputs
         << ", own " << fp32_own_outputs << "; bf16 own vs cuDNN " << bf16_own_outputs << "\n";

    // fp32: both forwards reproduce the reference gradient, and each other.
    EXPECT_LT(fp32_cudnn_error, 5.0e-3f);
    EXPECT_LT(fp32_own_error, 5.0e-3f);
    EXPECT_LT(fp32_own_vs_cudnn, 1.0e-3f);

    // bf16: same computation, same rounding points; must agree with cuDNN's.
    EXPECT_LT(bf16_own_vs_cudnn, 1.0e-2f);

    // Running statistics: inference after the one training forward matches the
    // reference in fp32 (measured ~1e-5; the running variance keeps cuDNN's
    // sample-variance convention, a factor rows/(rows-1) inside the momentum
    // term, and the reference the population one).
    EXPECT_LT(fp32_cudnn_outputs, 1.0e-3f);
    EXPECT_LT(fp32_own_outputs, 1.0e-3f);
    EXPECT_LT(bf16_own_outputs, 2.0e-2f);
}

// Max pooling on CUDA has two rungs: cuDNN's pooling, and the library's own
// forward that saves a one-byte argmax per output for a gather backward
// (MaxPoolingRung). A conv + ReLU feeding a 3x3 / stride 2 / pad 1 max pool -
// overlapping windows and padding, as in the ResNet stem - checks both rungs'
// gradients against the CPU reference in fp32, their agreement in bf16, and
// the own forward's outputs.
TEST_F(GpuComparison, MaxPoolingGradientPerRung)
{
    constexpr Index samples_number = 8;
    const Shape input_shape{4, 4, 16};

    TabularDataset dataset(samples_number, input_shape, Shape{1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    Tensor4 inputs(samples_number, 4, 4, 16);
    inputs.setRandom();

    const auto build_network = [&](NeuralNetwork& network)
    {
        network.add_layer(make_unique<Convolutional>(
                              input_shape, Shape{3, 3, 16, 32}, "ReLU",
                              Shape{1, 1}, "Same", false, "conv"),
                          {-1});
        network.add_layer(make_unique<Pooling>(Shape{4, 4, 32}, Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                                               "MaxPooling", "pool"),
                          {0});
        network.add_layer(make_unique<Flatten>(Shape{2, 2, 32}), {1});
        network.add_layer(make_unique<opennn::Dense>(Shape{128}, Shape{1}, "Identity"), {2});
        network.compile();
    };

    Configuration::instance().set(Device::CPU, Type::FP32);
    NeuralNetwork cpu_network;
    build_network(cpu_network);
    cpu_network.set_parameters_random();
    const VectorR parameters = read_host_parameters(cpu_network);

    Loss cpu_loss(&cpu_network, &dataset);
    cpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);
    const MatrixR cpu_outputs = cpu_network.calculate_outputs(inputs);

    struct RestoreRung
    {
        ~RestoreRung() { device::set_rung(device::MaxPoolingRung::Auto); }
    } restore;

    struct Run { VectorR gradient; MatrixR outputs; };
    const auto gpu_run = [&](Type type, device::MaxPoolingRung rung)
    {
        device::set_rung(rung);
        Configuration::instance().set(Device::CUDA, type);
        NeuralNetwork gpu_network;
        build_network(gpu_network);
        gpu_network.set_parameters(parameters);
        Loss gpu_loss(&gpu_network, &dataset);
        gpu_loss.set_error(Loss::Error::MeanSquaredError);
        Run run;
        run.gradient = calculate_gradient(gpu_loss);
        run.outputs = gpu_network.calculate_outputs(inputs);
        return run;
    };

    const Run fp32_cudnn = gpu_run(Type::FP32, device::MaxPoolingRung::Cudnn);
    const Run fp32_own   = gpu_run(Type::FP32, device::MaxPoolingRung::OwnKernel);
    const Run bf16_cudnn = gpu_run(Type::BF16, device::MaxPoolingRung::Cudnn);
    const Run bf16_own   = gpu_run(Type::BF16, device::MaxPoolingRung::OwnKernel);

    for (const Run* run : {&fp32_cudnn, &fp32_own, &bf16_cudnn, &bf16_own})
    {
        ASSERT_EQ(cpu_gradient.size(), run->gradient.size());
        ASSERT_EQ(cpu_outputs.rows(), run->outputs.rows());
    }

    const float fp32_cudnn_error  = relative_difference(cpu_gradient, fp32_cudnn.gradient);
    const float fp32_own_error    = relative_difference(cpu_gradient, fp32_own.gradient);
    const float bf16_own_vs_cudnn = relative_difference(bf16_cudnn.gradient, bf16_own.gradient);
    const float fp32_own_outputs  = relative_difference(cpu_outputs, fp32_own.outputs);
    const float bf16_own_outputs  = relative_difference(bf16_cudnn.outputs, bf16_own.outputs);

    cout << "max-pooling rungs - fp32 gradient vs reference: cuDNN " << fp32_cudnn_error
         << ", own " << fp32_own_error << "; bf16 gradient own vs cuDNN " << bf16_own_vs_cudnn
         << "; outputs: fp32 own vs reference " << fp32_own_outputs
         << ", bf16 own vs cuDNN " << bf16_own_outputs << "\n";

    EXPECT_LT(fp32_cudnn_error, 5.0e-3f);
    EXPECT_LT(fp32_own_error, 5.0e-3f);
    EXPECT_LT(bf16_own_vs_cudnn, 1.0e-2f);
    EXPECT_LT(fp32_own_outputs, 1.0e-3f);
    EXPECT_LT(bf16_own_outputs, 1.0e-2f);
}

// The real ResNet builder, small: a stem, a projection-skip bottleneck block and
// an identity bottleneck block per stage, two stages, so every block wiring the
// residual-join fusion (BackPropagation::plan_delta_addends -> conv dgrad + ADD)
// meets in ResNet-50 is present. Whole-network gradient, GPU vs CPU reference,
// in fp32 (exactness) and bf16 (precision band).
TEST_F(GpuComparison, ResNetBottleneckGradient)
{
    constexpr Index samples_number = 8;
    const Shape input_shape{16, 16, 3};
    const Index classes_number = 4;

    TabularDataset dataset(samples_number, input_shape, Shape{classes_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    const auto build = [&]() -> unique_ptr<NeuralNetwork>
    {
        return make_unique<ResNet>(input_shape, vector<Index>{2, 2}, Shape{8, 16},
                                   Shape{classes_number}, true);
    };

    Configuration::instance().set(Device::CPU, Type::FP32);
    unique_ptr<NeuralNetwork> cpu_network = build();
    cpu_network->set_parameters_random();
    const VectorR parameters = read_host_parameters(*cpu_network);

    Loss cpu_loss(cpu_network.get(), &dataset);
    cpu_loss.set_error(Loss::Error::CrossEntropy);
    const VectorR cpu_gradient = calculate_gradient(cpu_loss);

    const auto gpu_gradient_in = [&](Type type)
    {
        Configuration::instance().set(Device::CUDA, type);
        unique_ptr<NeuralNetwork> gpu_network = build();
        gpu_network->set_parameters(parameters);
        Loss gpu_loss(gpu_network.get(), &dataset);
        gpu_loss.set_error(Loss::Error::CrossEntropy);
        return calculate_gradient(gpu_loss);
    };

    const VectorR fp32 = gpu_gradient_in(Type::FP32);
    const VectorR bf16 = gpu_gradient_in(Type::BF16);

    ASSERT_EQ(cpu_gradient.size(), fp32.size());
    ASSERT_EQ(cpu_gradient.size(), bf16.size());

    const float fp32_error = relative_difference(cpu_gradient, fp32);
    const float bf16_error = relative_difference(cpu_gradient, bf16);

    // The fp32 bound cannot be a fixed constant. OpenNN's GPU fp32 convolutions
    // run on tensor cores at TF32, whose 10-bit mantissa is ~1e-3 relative, and
    // an untrained bottleneck ResNet amplifies that hard: perturbing the CPU
    // parameters by a relative 1e-3 and re-running entirely on the CPU moves
    // this gradient by 0.16, while the CPU/GPU split is 0.13 (RTX 5070 Ti,
    // sm_120). A fixed 5e-3 bar passed only where cuDNN happened to pick
    // true-fp32 engines for these shapes; it fails on hardware that picks
    // TF32 ones, without anything being wrong.
    //
    // So calibrate against the network instead: rerun the CPU gradient with the
    // parameters rounded to TF32 resolution, and require the GPU to be no
    // further from the CPU than that. A genuinely wrong kernel lands far
    // outside this envelope -- with the residual-join fold disabled, or any
    // batch-norm rung forced, the split stays at 0.13, i.e. inside it.
    unique_ptr<NeuralNetwork> tf32_network = build();
    VectorR tf32_parameters = parameters;
    for (Index i = 0; i < tf32_parameters.size(); ++i)
    {
        uint32_t bits;
        memcpy(&bits, &tf32_parameters[i], sizeof(bits));
        bits &= 0xffffe000u;                       // keep TF32's 10 mantissa bits
        memcpy(&tf32_parameters[i], &bits, sizeof(bits));
    }
    Configuration::instance().set(Device::CPU, Type::FP32);
    tf32_network->set_parameters(tf32_parameters);
    Loss tf32_loss(tf32_network.get(), &dataset);
    tf32_loss.set_error(Loss::Error::CrossEntropy);
    const float tf32_envelope =
        relative_difference(cpu_gradient, calculate_gradient(tf32_loss));

    const float fp32_bound = max(5.0e-3f, 2.0f * tf32_envelope);

    cout << "ResNet bottleneck gradient vs fp32 reference: fp32 GPU " << fp32_error
         << ", bf16 GPU " << bf16_error << ", TF32 envelope " << tf32_envelope
         << ", fp32 bound " << fp32_bound << "\n";

    // bf16 carries this small random network's end-to-end rounding (measured
    // 0.16-0.38 depending on the draw and on GPU reduction order), so its bound
    // is a sanity check, not a precision claim.
    EXPECT_LT(fp32_error, fp32_bound);
    EXPECT_LT(bf16_error, 5.0e-1f);
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
    device::copy_async(input_buffer.data(), inputs.data(), input_bytes,
                       device::CopyKind::HostToDevice);
    device::synchronize();

    const TensorView input_view(input_buffer.data(),
                                Shape{samples_number, height, width, channels},
                                Type::FP32, Device::CUDA);

    ForwardPropagation forward_propagation(samples_number, &network);

    const auto read_outputs = [](const TensorView& outputs)
    {
        vector<float> host(size_t(outputs.size()));
        device::synchronize();
        copy_device_to_host_float(outputs.get_data(), outputs.get_type(), outputs.size(),
                                  host.data(), device::get_compute_stream());
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
    device::copy_async(input_buffer.data(), inputs.data(), input_bytes,
                       device::CopyKind::HostToDevice);
    Buffer second_input_buffer;
    second_input_buffer.resize_bytes(input_bytes, Device::CUDA);
    device::copy_async(second_input_buffer.data(), inputs.data(), input_bytes,
                       device::CopyKind::HostToDevice);
    device::synchronize();

    const TensorView input_view(input_buffer.data(),
                                Shape{samples_number, height, width, channels},
                                Type::FP32, Device::CUDA);
    const TensorView second_input_view(second_input_buffer.data(),
                                       Shape{samples_number, height, width, channels},
                                       Type::FP32, Device::CUDA);

    ForwardPropagation forward_propagation(samples_number, &network);

    const auto read_outputs = [](const TensorView& outputs)
    {
        vector<float> host(size_t(outputs.size()));
        device::synchronize();
        copy_device_to_host_float(outputs.get_data(), outputs.get_type(), outputs.size(),
                                  host.data(), device::get_compute_stream());
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

TEST_F(GpuComparison, RecurrentExecutionStateIsPropagationOwned)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    NeuralNetwork network;
    network.add_layer(make_unique<Recurrent>(Shape{4, 3}, Shape{5}, "Tanh"));
    network.compile();
    network.set_parameters_random();

    Tensor3 inputs(2, 4, 3);
    inputs.setRandom();
    const vector<TensorView> input_views = {
        TensorView(inputs.data(), {2, 4, 3})
    };

    ForwardPropagation first(2, &network);
    ForwardPropagation second(2, &network);

    network.forward_propagate(input_views, first, true);
    network.forward_propagate(input_views, second, true);

    ASSERT_EQ(first.layer_state_storage.size(), 1);
    ASSERT_EQ(second.layer_state_storage.size(), 1);
    EXPECT_FALSE(first.layer_state_storage[0].empty());
    EXPECT_FALSE(second.layer_state_storage[0].empty());
    EXPECT_EQ(first.layer_state_storage[0].get_device(), Device::CUDA);
    EXPECT_NE(first.layer_state_storage[0].data(),
              second.layer_state_storage[0].data());
}

TEST_F(GpuComparison, RnnDescriptorCacheSupportsConcurrentMixedBatches)
{
    constexpr Index time_steps = 4;
    constexpr Index input_features = 3;
    constexpr Index output_features = 5;

    Tensor3 first_inputs(2, time_steps, input_features);
    Tensor3 second_inputs(5, time_steps, input_features);
    first_inputs.setRandom();
    second_inputs.setRandom();

    {
        SCOPED_TRACE("Recurrent");
        const auto build_network = [&](NeuralNetwork& network)
        {
            network.add_layer(make_unique<Recurrent>(
                                  Shape{time_steps, input_features},
                                  Shape{output_features}, "Tanh"),
                              {-1});
            network.compile();
        };
        expect_concurrent_gpu_outputs(first_inputs, second_inputs, build_network);
    }

    {
        SCOPED_TRACE("LongShortTermMemory");
        const auto build_network = [&](NeuralNetwork& network)
        {
            network.add_layer(make_unique<LongShortTermMemory>(
                                  Shape{time_steps, input_features},
                                  Shape{output_features}, "Tanh", "Sigmoid"),
                              {-1});
            network.compile();
        };
        expect_concurrent_gpu_outputs(first_inputs, second_inputs, build_network);
    }
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

TEST_F(GpuComparison, ForecastingRecurrentAndLstmBf16Forward)
{
    constexpr Index samples_number = 7;
    constexpr Index past = 5;
    constexpr Index features = 3;

    Tensor3 inputs(samples_number, past, features);
    inputs.setRandom();

    const auto compare = [&](bool lstm)
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        unique_ptr<NeuralNetwork> cpu = lstm
            ? unique_ptr<NeuralNetwork>(make_unique<ForecastingLstmNetwork>(
                  Shape{past, features}, Shape{6, 5}, Shape{1}))
            : unique_ptr<NeuralNetwork>(make_unique<ForecastingNetwork>(
                  Shape{past, features}, Shape{6, 5}, Shape{1}));
        cpu->set_parameters_random();
        const VectorR parameters = read_host_parameters(*cpu);
        const MatrixR reference = cpu->calculate_outputs(inputs);

        Configuration::instance().set(Device::CUDA, Type::BF16);
        unique_ptr<NeuralNetwork> gpu = lstm
            ? unique_ptr<NeuralNetwork>(make_unique<ForecastingLstmNetwork>(
                  Shape{past, features}, Shape{6, 5}, Shape{1}))
            : unique_ptr<NeuralNetwork>(make_unique<ForecastingNetwork>(
                  Shape{past, features}, Shape{6, 5}, Shape{1}));
        gpu->set_parameters(parameters);
        const MatrixR actual = gpu->calculate_outputs(inputs);

        EXPECT_LT(relative_difference(reference, actual), 5.0e-2f);
    };

    compare(false);
    compare(true);
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

TEST_F(GpuComparison, ForecastingRecurrentAndLstmBf16Gradient)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    set_seed(17);
    TimeSeriesDataset dataset(30, {2}, {1});
    dataset.set_data_random();
    dataset.set_past_time_steps(5);
    dataset.set_future_time_steps(1);
    dataset.set_sample_roles("Training");

    const auto compare = [&](bool lstm)
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        unique_ptr<NeuralNetwork> cpu = lstm
            ? unique_ptr<NeuralNetwork>(make_unique<ForecastingLstmNetwork>(
                  dataset.get_input_shape(), Shape{6, 5}, dataset.get_target_shape()))
            : unique_ptr<NeuralNetwork>(make_unique<ForecastingNetwork>(
                  dataset.get_input_shape(), Shape{6, 5}, dataset.get_target_shape()));
        cpu->set_parameters_random();
        const VectorR parameters = read_host_parameters(*cpu);
        Loss cpu_loss(cpu.get(), &dataset);
        cpu_loss.set_error(Loss::Error::MeanSquaredError);
        const VectorR reference = calculate_gradient(cpu_loss);

        Configuration::instance().set(Device::CUDA, Type::BF16);
        unique_ptr<NeuralNetwork> gpu = lstm
            ? unique_ptr<NeuralNetwork>(make_unique<ForecastingLstmNetwork>(
                  dataset.get_input_shape(), Shape{6, 5}, dataset.get_target_shape()))
            : unique_ptr<NeuralNetwork>(make_unique<ForecastingNetwork>(
                  dataset.get_input_shape(), Shape{6, 5}, dataset.get_target_shape()));
        gpu->set_parameters(parameters);
        Loss gpu_loss(gpu.get(), &dataset);
        gpu_loss.set_error(Loss::Error::MeanSquaredError);
        const VectorR actual = calculate_gradient(gpu_loss);

        ASSERT_EQ(reference.size(), actual.size());
        EXPECT_LT(relative_difference(reference, actual), 1.0e-1f);
    };

    compare(false);
    compare(true);
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

// The whole encoder-decoder training gradient: CPU against the numerical
// gradient, and GPU against CPU (the residual fan-outs fold one consumer's
// delta into the other's GEMM on both devices, the GPU through cuBLASLt's C).
static void write_transformer_gradient_corpus(const string& file_path)
{
    string content;
    for (Index sample = 0; sample < 4; ++sample)
    {
        for (Index token = 0; token < 4; ++token)
            content += "w" + to_string((sample * 3 + token * 5) % 9) + (token + 1 < 4 ? " " : "");
        content += "\t";
        for (Index token = 0; token < 3; ++token)
            content += "t" + to_string((sample * 2 + token * 3) % 7) + (token + 1 < 3 ? " " : "");
        content += "\n";
    }
    ofstream out(file_path);
    out << content;
}

static VectorR transformer_training_gradient(Device device, const string& corpus,
                                             const VectorR* parameters, VectorR* parameters_out,
                                             VectorR* numerical_out)
{
    Configuration::instance().set(device, Type::FP32);
    set_seed(7);
    LanguageDataset dataset(corpus);
    dataset.set_display(false);
    dataset.split_samples(1.0f, 0.0f, 0.0f);

    Transformer transformer(dataset.get_shape("Input")[0], dataset.get_shape("Decoder")[0],
                            dataset.get_input_vocabulary_size(), dataset.get_target_vocabulary_size(),
                            8, 2, 16, 1);
    transformer.set_dropout_rate(0.0f);
    transformer.set_attention_sdpa_min_sequence_length(1 << 20);   // unfused attention on both devices
    if (parameters) transformer.set_parameters(*parameters);
    else            transformer.set_parameters_random();
    if (parameters_out) *parameters_out = read_host_parameters(transformer);

    Loss loss(&transformer, &dataset);
    loss.set_error(Loss::Error::CrossEntropy3d);

    if (numerical_out) *numerical_out = calculate_numerical_gradient(loss);
    return calculate_gradient(loss);
}

TEST_F(GpuComparison, TransformerTrainingGradient)
{
    const string corpus = (filesystem::temp_directory_path() / "opennn_gpu_transformer_gradient.txt").string();
    write_transformer_gradient_corpus(corpus);

    VectorR parameters, numerical;
    const VectorR cpu_gradient = transformer_training_gradient(Device::CPU, corpus, nullptr, &parameters, &numerical);
    EXPECT_LT((cpu_gradient - numerical).array().abs().maxCoeff(), 2.0e-3f)
        << "CPU transformer gradient differs from the numerical one";

    const VectorR gpu_gradient = transformer_training_gradient(Device::CUDA, corpus, &parameters, nullptr, nullptr);
    ASSERT_EQ(gpu_gradient.size(), cpu_gradient.size());
    EXPECT_LT(relative_difference(cpu_gradient, gpu_gradient), 1.0e-3f);

    filesystem::remove(corpus);
}

// FlashAttention-2 against cuDNN's fused attention, over a transformer whose
// samples are of different lengths: every encoder self-attention and every
// cross-attention in it runs the FA2 rung over a padded batch, and every
// decoder self-attention stays on cuDNN, because a causal mask and a padded
// batch are the one combination FA2 anchors differently (see
// core/cuda/flash_attention.cuh). Two bf16 attention kernels do not agree to
// the last bit, so what this checks is a gradient of the same shape and size,
// which a mask read the wrong way, a stride crossed or a log-sum-exp the
// backward could not use would all miss by far more.
static void write_ragged_transformer_corpus(const string& file_path)
{
    ofstream out(file_path);
    for (Index sample = 0; sample < 4; ++sample)
    {
        const Index input_tokens = 4 - sample % 3;      // 4, 3, 2, 4
        const Index target_tokens = 3 - sample % 2;     // 3, 2, 3, 2

        for (Index token = 0; token < input_tokens; ++token)
            out << "w" << (sample * 3 + token * 5) % 9 << (token + 1 < input_tokens ? " " : "");
        out << "\t";
        for (Index token = 0; token < target_tokens; ++token)
            out << "t" << (sample * 2 + token * 3) % 7 << (token + 1 < target_tokens ? " " : "");
        out << "\n";
    }
}

static VectorR fused_transformer_gradient(const string& corpus, device::AttentionRung rung,
                                          const VectorR* parameters, VectorR* parameters_out)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    set_seed(7);
    LanguageDataset dataset(corpus);
    dataset.set_display(false);
    dataset.split_samples(1.0f, 0.0f, 0.0f);

    // Head dimension 32 is one FA2 ships a kernel for; 64 over two heads is how
    // this transformer gets there.
    Transformer transformer(dataset.get_shape("Input")[0], dataset.get_shape("Decoder")[0],
                            dataset.get_input_vocabulary_size(), dataset.get_target_vocabulary_size(),
                            64, 2, 16, 1);
    transformer.set_dropout_rate(0.0f);
    transformer.set_attention_sdpa_min_sequence_length(1);   // fused attention on both rungs

    if (parameters) transformer.set_parameters(*parameters);
    else            transformer.set_parameters_random();
    if (parameters_out) *parameters_out = read_host_parameters(transformer);

    Loss loss(&transformer, &dataset);
    loss.set_error(Loss::Error::CrossEntropy3d);

    // Restored however this leaves, so a failure here does not pin the rung for
    // every test that runs after it.
    struct RestoreRung
    {
        device::AttentionRung previous = device::rung<device::AttentionRung>();
        ~RestoreRung() { device::set_rung(previous); }
    } restore;

    device::set_rung(rung);

    return calculate_gradient(loss);
}

TEST_F(GpuComparison, FlashAttentionRungMatchesCudnnAttention)
{
    // Whether this build has a kernel here at all: what applies() reads of a
    // problem this small is the head dimension, the device and the mask.
    const flash_attention::Problem probe{
        .batch = 1, .heads = 1,
        .query_sequence_length = 1, .source_sequence_length = 1,
        .head_dimension = 32,
        .causal = false, .scale = 1.0f
    };

    if (!flash_attention::applies(probe))
        GTEST_SKIP() << "this build has no FlashAttention-2 kernel for this device";

    const string corpus = (filesystem::temp_directory_path() / "opennn_flash_attention_rung.txt").string();
    write_ragged_transformer_corpus(corpus);

    VectorR parameters;
    const Index before = flash_attention::call_count();
    const VectorR flash_gradient = fused_transformer_gradient(corpus, device::AttentionRung::Auto,
                                                              nullptr, &parameters);
    EXPECT_GT(flash_attention::call_count(), before)
        << "the FlashAttention rung never ran, so this test compared cuDNN with itself";

    const VectorR cudnn_gradient = fused_transformer_gradient(corpus, device::AttentionRung::CudnnGraph,
                                                              &parameters, nullptr);

    ASSERT_EQ(flash_gradient.size(), cudnn_gradient.size());
    EXPECT_LT(relative_difference(cudnn_gradient, flash_gradient), 5.0e-2f);

    filesystem::remove(corpus);
}

// The dense single-output backward does the input delta, the weight gradient
// and the producing ReLU's derivative in one pass. Correctness is covered by the
// gradient tests; what this one guards is that the fast path is still taken -
// dropping it costs 5-9% on the HIGGS benchmark and changes no result, so
// nothing else would notice.
TEST_F(GpuComparison, DenseSingleOutputBackwardFoldsProducerRelu)
{
    const Index samples_number = 16;
    const Index inputs_number = 28;
    const Index hidden_number = 1024;   // 32 lanes x whole 16-byte vectors

    Configuration::instance().set(Device::CUDA, Type::FP32);

    TabularDataset dataset(samples_number, {inputs_number}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "ReLU"));
    network.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{1}, "Sigmoid"));
    network.compile();
    network.set_parameters_random();

    const auto* output_layer = dynamic_cast<const opennn::Dense*>(network.get_layers()[1].get());
    ASSERT_NE(output_layer, nullptr);
    EXPECT_TRUE(output_layer->single_output_relu_fusion_wired())
        << "the single-output layer no longer absorbs the ReLU backward of its producer";

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    Batch batch(samples_number, &dataset, network.get_config());
    batch.fill(dataset.get_sample_indices("Training"),
               dataset.get_feature_indices("Input"),
               dataset.get_feature_indices("Decoder"),
               dataset.get_feature_indices("Target"));
    batch.upload_to_device_batch_async(batch, device::get_transfer_stream());
    batch.wait_h2d_complete();

    ForwardPropagation forward_propagation(samples_number, &network);
    BackPropagation back_propagation(samples_number, loss);

    network.forward_propagate(batch.get_inputs(), forward_propagation, true);
    loss.back_propagate(batch, forward_propagation, back_propagation);

    // The consumer reports through the same per-layer flag the DReLU epilogue
    // uses, and the producer's activation backward reads it to stay out.
    EXPECT_NE(forward_propagation.drelu_fused_by_layer[0], 0)
        << "the one-pass backward ran but did not report the fold, so the ReLU "
           "backward of layer 0 ran a second time over the same tensor";

    // A layer with more than one output keeps the general path.
    NeuralNetwork wide;
    wide.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{hidden_number}, "ReLU"));
    wide.add_layer(make_unique<opennn::Dense>(Shape{hidden_number}, Shape{2}, "Sigmoid"));
    wide.compile();
    const auto* wide_output = dynamic_cast<const opennn::Dense*>(wide.get_layers()[1].get());
    ASSERT_NE(wide_output, nullptr);
    EXPECT_FALSE(wide_output->single_output_relu_fusion_wired());
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
    network.add_layer(std::move(attention));
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
        device::copy_async(host.data(), outputs.get_data(),
                           outputs.size() * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost,
                           device::get_compute_stream());
        device::synchronize(device::get_compute_stream());
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
    gpu_network.add_layer(std::move(gpu_attention));
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

TEST_F(GpuComparison, SdpaDropoutBackwardUsesForwardState)
{
    if (!AttentionOperator::sdpa_supported(Type::FP32, Device::CUDA))
        GTEST_SKIP() << "SDPA is not available in this build.";

    constexpr Index samples_number = 4;
    constexpr Index sequence_length = 64;
    constexpr Index embedding_dimension = 32;
    constexpr Index heads_number = 2;

    Configuration::instance().set(Device::CUDA, Type::FP32);

    TabularDataset dataset(samples_number,
                           Shape{sequence_length, embedding_dimension},
                           Shape{sequence_length * embedding_dimension});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    auto attention = make_unique<MultiHeadAttention>(
        Shape{sequence_length, embedding_dimension}, heads_number);
    attention->set_sdpa_min_sequence_length(1);
    attention->set_dropout_rate(0.25f);

    NeuralNetwork network;
    network.add_layer(std::move(attention));
    network.add_layer(make_unique<Flatten>(network.get_output_shape()));
    network.compile();
    network.set_parameters_random();

    ASSERT_TRUE(static_cast<MultiHeadAttention*>(network.get_layer(0).get())
                    ->should_use_sdpa());

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    EXPECT_TRUE(gradient.allFinite());
    EXPECT_GT(gradient.squaredNorm(), 0.0f);
}

// The padding an activation scan cannot see. An Embedding zeroes the row of a
// padding token, which is what lets the scan behind the fused path recover the
// sequence lengths -- but only for an attention layer reading the Embedding
// directly. BERT normalizes first, and a normalization turns a zero row into its
// own bias, so from the first encoder block onwards the only surviving record of
// where a sequence ends is the length the Embedding exports. The normalization
// here puts the attention layer in that position. Masking with those exported
// lengths must land on the same numbers as the hand-written mask that consumes
// them today, in the gradients as well as the outputs, because the backward
// graph masks with the same two tensors the forward filled.
TEST_F(GpuComparison, SdpaAttentionMatchesUnfusedOnExportedValidLengths)
{
    if (!AttentionOperator::sdpa_supported(Type::FP32, Device::CUDA))
        GTEST_SKIP() << "SDPA is not available in this build.";

    const Index samples_number = 4;
    const Index sequence_length = 16;
    const Index embedding_dimension = 32;
    const Index heads_number = 2;
    const Index vocabulary_size = 24;

    Configuration::instance().set(Device::CPU, Type::FP32);

    TabularDataset dataset(samples_number, Shape{sequence_length},
                           {sequence_length * embedding_dimension});
    set_seed(13);
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    // Token id 0 is padding, so a shorter prefix of nonzero ids is a shorter
    // sequence. One full row keeps an unpadded sample in the batch and one row
    // of length 1 exercises the shortest sequence the mask has to survive.
    const std::array<Index, 4> valid_lengths{sequence_length, 11, 5, 1};
    MatrixR data = dataset.get_data();
    for (Index sample = 0; sample < samples_number; ++sample)
        for (Index position = 0; position < sequence_length; ++position)
            data(sample, position) = position < valid_lengths[size_t(sample)]
                ? float(1 + (sample * sequence_length + position) % (vocabulary_size - 1))
                : 0.0f;
    dataset.set_data(std::move(data));

    const auto build = [&](const bool fused)
    {
        auto network = make_unique<NeuralNetwork>();

        auto embedding = make_unique<Embedding>(Shape{vocabulary_size, sequence_length},
                                                embedding_dimension, "embedding");
        embedding->set_add_positional_encoding(true);
        embedding->set_export_valid_lengths(true);
        network->add_layer(std::move(embedding), {-1});

        network->add_layer(make_unique<Normalization3d>(
                               Shape{sequence_length, embedding_dimension}, "normalization"),
                           {0});

        auto attention = make_unique<MultiHeadAttention>(
            Shape{sequence_length, embedding_dimension}, heads_number);
        attention->set_sdpa_min_sequence_length(1);
        attention->set_sdpa_auto(fused);
        network->add_layer(std::move(attention), {1});

        network->add_layer(make_unique<Flatten>(network->get_output_shape()));
        network->compile();
        return network;
    };

    const auto gradient_of = [&](NeuralNetwork& network)
    {
        Loss loss(&network, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        return calculate_gradient(loss);
    };

    const auto cpu_network = build(false);
    cpu_network->set_parameters_random();

    // A normalization starts at gamma = 1, beta = 0, which maps a zero row to
    // zero and so leaves the padding visible to the scan after all. Training
    // moves the shift off zero, and that is the state this test needs: with a
    // nonzero beta the padded rows are ordinary values and the exported lengths
    // are the only thing that still knows they are padding.
    cpu_network->get_layer(1)->get_parameter_views()[1].as_vector().setConstant(0.25f);

    const VectorR parameters = read_host_parameters(*cpu_network);
    const VectorR cpu_gradient = gradient_of(*cpu_network);

    Configuration::instance().set(Device::CUDA, Type::FP32);

    const auto unfused_network = build(false);
    unfused_network->set_parameters(parameters);
    ASSERT_FALSE(static_cast<MultiHeadAttention*>(unfused_network->get_layer(2).get())->should_use_sdpa());

    const auto fused_network = build(true);
    fused_network->set_parameters(parameters);
    ASSERT_TRUE(static_cast<MultiHeadAttention*>(fused_network->get_layer(2).get())->should_use_sdpa());

    const VectorR unfused_gradient = gradient_of(*unfused_network);
    const VectorR fused_gradient   = gradient_of(*fused_network);

    ASSERT_EQ(cpu_gradient.size(), fused_gradient.size());

    // Both GPU paths are held to the tolerance SdpaAttentionBackwardGradient
    // already uses, which is set by the BF16 cast cuDNN's fused attention forces
    // on an FP32 network, not by the mask. Ignoring the exported lengths would
    // leave the fused path attending over the padding -- and the positional
    // encoding makes those rows nonzero, so nothing else would mask them -- for
    // a disagreement of order one rather than of order the tolerance.
    EXPECT_LT(relative_difference(cpu_gradient, unfused_gradient), 2.0e-2f);
    EXPECT_LT(relative_difference(cpu_gradient, fused_gradient), 2.0e-2f);
}

// zero_padded_queries asks attention for exactly-zero output rows at padded
// query positions, and vetoes fused attention to get them, because cuDNN does
// not document what it writes outside the valid region. The only reason anyone
// wanted those zeros was that average pooling used to recover the sequence
// length by looking for zero rows. It reads the exported lengths now, so the
// padded rows are read by nobody and what cuDNN leaves in them stops being a
// question that has to be answered.
//
// This is the network that made the demand -- an Embedding, attention, and an
// averaging pool -- with the demand withdrawn. Fused and unfused have to reach
// the same gradient through it. Note what is NOT claimed: that the two agree
// row by row inside attention. They do not, and need not. Even unfused, a
// padded query row is a weighted average of the valid values rather than zero,
// because zeroing it is exactly the thing this network no longer asks for.
TEST_F(GpuComparison, SdpaMatchesUnfusedThroughAveragePoolingOnPaddedBatches)
{
    if (!AttentionOperator::sdpa_supported(Type::FP32, Device::CUDA))
        GTEST_SKIP() << "SDPA is not available in this build.";

    const Index samples_number = 4;
    const Index sequence_length = 16;
    const Index embedding_dimension = 32;
    const Index heads_number = 2;
    const Index vocabulary_size = 24;
    const Index targets_number = 2;

    Configuration::instance().set(Device::CPU, Type::FP32);

    TabularDataset dataset(samples_number, Shape{sequence_length}, Shape{targets_number});
    set_seed(29);
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    const std::array<Index, 4> valid_lengths{sequence_length, 11, 5, 1};
    MatrixR data = dataset.get_data();
    for (Index sample = 0; sample < samples_number; ++sample)
        for (Index position = 0; position < sequence_length; ++position)
            data(sample, position) = position < valid_lengths[size_t(sample)]
                ? float(1 + (sample * sequence_length + position) % (vocabulary_size - 1))
                : 0.0f;
    dataset.set_data(std::move(data));

    const auto build = [&](const bool fused)
    {
        auto network = make_unique<NeuralNetwork>();

        auto embedding = make_unique<Embedding>(Shape{vocabulary_size, sequence_length},
                                                embedding_dimension, "embedding");
        embedding->set_add_positional_encoding(true);
        embedding->set_export_valid_lengths(true);
        network->add_layer(std::move(embedding), {-1});

        auto attention = make_unique<MultiHeadAttention>(
            Shape{sequence_length, embedding_dimension}, heads_number);
        attention->set_sdpa_min_sequence_length(1);
        attention->set_sdpa_auto(fused);
        network->add_layer(std::move(attention), {0});

        network->add_layer(make_unique<Pooling3d>(Shape{sequence_length, embedding_dimension},
                                                  PoolingMethod::AveragePooling, "pool"), {1});

        network->add_layer(make_unique<opennn::Dense>(network->get_layer(2)->get_output_shape(),
                                                      dataset.get_target_shape()), {2});
        network->compile();
        return network;
    };

    const auto gradient_of = [&](NeuralNetwork& network)
    {
        Loss loss(&network, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        return calculate_gradient(loss);
    };

    const auto cpu_network = build(false);
    cpu_network->set_parameters_random();

    const VectorR parameters = read_host_parameters(*cpu_network);
    const VectorR cpu_gradient = gradient_of(*cpu_network);

    Configuration::instance().set(Device::CUDA, Type::FP32);

    const auto unfused_network = build(false);
    unfused_network->set_parameters(parameters);
    ASSERT_FALSE(static_cast<MultiHeadAttention*>(unfused_network->get_layer(1).get())->should_use_sdpa());

    const auto fused_network = build(true);
    fused_network->set_parameters(parameters);

    // The veto is what this test is about: with no zero_padded_queries asked
    // for, a padded batch is allowed to reach cuDNN's fused attention at all.
    ASSERT_TRUE(static_cast<MultiHeadAttention*>(fused_network->get_layer(1).get())->should_use_sdpa());

    const VectorR unfused_gradient = gradient_of(*unfused_network);
    const VectorR fused_gradient   = gradient_of(*fused_network);

    ASSERT_EQ(cpu_gradient.size(), fused_gradient.size());

    EXPECT_LT(relative_difference(cpu_gradient, unfused_gradient), 2.0e-2f);
    EXPECT_LT(relative_difference(cpu_gradient, fused_gradient), 2.0e-2f);
}

#endif
