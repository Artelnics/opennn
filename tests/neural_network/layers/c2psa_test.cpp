#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include "opennn/neural_network/layers/c2psa_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"

using namespace opennn;

static constexpr Index H_GRID  = 2;
static constexpr Index W_GRID  = 2;
static constexpr Index CHAN    = 8;
static constexpr Index TARGETS = 1;
static constexpr Index SAMPLES = 3;

struct C2PSANet
{
    TabularDataset dataset{SAMPLES, Shape{H_GRID, W_GRID, CHAN}, Shape{TARGETS}};
    NeuralNetwork  nn;

    C2PSANet()
    {
        dataset.set_data_random();
        dataset.set_sample_roles("Training");

        nn.add_layer(make_unique<C2PSA>(Shape{H_GRID, W_GRID, CHAN}, "c2psa"));

        const Shape c2psa_out = nn.get_layer(0)->get_output_shape();
        nn.add_layer(make_unique<Flatten>(c2psa_out));

        const Shape flat_out = nn.get_layer(1)->get_output_shape();
        nn.add_layer(make_unique<opennn::Dense>(flat_out, Shape{TARGETS}));

        nn.compile();

        // Without this every parameter is zero, and a C2PSA with zero weights
        // emits zeros: Q, K and V are zero, so the attention gradient is zero
        // too and every check below compares zero against zero.
        nn.set_parameters_random();
    }

    unique_ptr<Loss> make_loss()
    {
        auto l = make_unique<Loss>(&nn, &dataset);
        l->set_error(Loss::Error::MeanSquaredError);
        return l;
    }
};

TEST(C2PSA, CpuGradientMatchesNumerical)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    C2PSANet net;
    auto loss = net.make_loss();

    const VectorR analytical = calculate_gradient(*loss);
    const VectorR numerical  = calculate_numerical_gradient(*loss);

    ASSERT_EQ(analytical.size(), numerical.size());
    const float max_diff = (analytical - numerical).array().abs().maxCoeff();
    EXPECT_LT(max_diff, 1e-3f)
        << "Max element-wise diff (CPU): " << max_diff;
}

TEST(C2PSA, GpuGradientMatchesNumerical)
{
    if (!opennn::device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);

    C2PSANet net;
    auto loss = net.make_loss();

    const VectorR analytical = calculate_gradient(*loss);
    const VectorR numerical  = calculate_numerical_gradient(*loss);

    ASSERT_EQ(analytical.size(), numerical.size());
    const float max_diff = (analytical - numerical).array().abs().maxCoeff();
    EXPECT_LT(max_diff, 1e-3f)
        << "Max element-wise diff (CUDA): " << max_diff;
}

TEST(C2PSA, CpuAndGpuForwardOutputsMatch)
{
    if (!opennn::device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    const Index batch_size = SAMPLES;

    Configuration::instance().set(Device::CPU, Type::FP32);

    C2PSANet cpu_net;
    const VectorR parameters = cpu_net.nn.get_parameters_map();
    const vector<Index> training_idx = cpu_net.dataset.get_sample_indices("Training");
    const vector<Index> input_idx    = cpu_net.dataset.get_feature_indices("Input");

    Batch batch(batch_size, &cpu_net.dataset, cpu_net.nn.get_config());
    batch.fill(training_idx, FeatureSelection{input_idx, {}, {}});
    ForwardPropagation fp(batch_size, &cpu_net.nn);
    cpu_net.nn.forward_propagate(batch.get_inputs(), fp, false);

    const TensorView out = fp.get_outputs();
    const Index n = out.size();
    vector<float> cpu_out(n);
    copy_n(out.as<float>(), n, cpu_out.data());

    Configuration::instance().set(Device::CUDA, Type::FP32);

    C2PSANet gpu_net;
    gpu_net.nn.set_parameters(parameters);

    Batch batch_gpu(batch_size, &cpu_net.dataset, gpu_net.nn.get_config());
    batch_gpu.fill(training_idx, FeatureSelection{input_idx, {}, {}});
    ForwardPropagation fp_gpu(batch_size, &gpu_net.nn);
    gpu_net.nn.forward_propagate(batch_gpu.get_inputs(), fp_gpu, false);
    const TensorView out_gpu = fp_gpu.get_outputs();

    vector<float> gpu_out(n);
#ifdef OPENNN_HAS_CUDA
    cudaMemcpy(gpu_out.data(), out_gpu.as<float>(), n * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    Configuration::instance().set(Device::CPU, Type::FP32);

    float max_diff = 0.0f;
    for (Index i = 0; i < n; ++i)
        max_diff = max(max_diff, abs(cpu_out[i] - gpu_out[i]));

    EXPECT_LT(max_diff, 1e-4f)
        << "Max CPU vs GPU forward output diff: " << max_diff;
}

TEST(C2PSA, GpuScratchIsPropagationOwned)
{
    if (!opennn::device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);

    C2PSANet net;
    auto loss = net.make_loss();

    ForwardPropagation first_forward(SAMPLES, &net.nn);
    ForwardPropagation second_forward(SAMPLES, &net.nn);
    BackPropagation first_backward(SAMPLES, *loss);
    BackPropagation second_backward(SAMPLES, *loss);

    const TensorView& first_forward_scratch =
        first_forward.slots[0][first_forward.slots[0].size() - 2];
    const TensorView& second_forward_scratch =
        second_forward.slots[0][second_forward.slots[0].size() - 2];
    const TensorView& first_backward_scratch = first_backward.slots[0].back();
    const TensorView& second_backward_scratch = second_backward.slots[0].back();

    ASSERT_FALSE(first_forward_scratch.empty());
    ASSERT_FALSE(first_backward_scratch.empty());
    EXPECT_NE(first_forward_scratch.get_data(), second_forward_scratch.get_data());
    EXPECT_NE(first_backward_scratch.get_data(), second_backward_scratch.get_data());
}

// The numerical checks above compare against an absolute 1e-3, and this network
// is small enough that its gradient sits near that -- loose enough to let a
// wrong constant factor on the attention gradient pass. The CPU backward is the
// reference for the GPU one here: same data, same weights, compared component
// by component relative to the size of the component, so an error in one block
// of the gradient cannot hide behind a larger block somewhere else.
TEST(C2PSA, CpuAndGpuGradientsMatch)
{
    if (!opennn::device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CPU, Type::FP32);

    C2PSANet cpu_net;
    const VectorR parameters = cpu_net.nn.get_parameters_map();
    auto cpu_loss = cpu_net.make_loss();
    const VectorR cpu_gradient = calculate_gradient(*cpu_loss);

    Configuration::instance().set(Device::CUDA, Type::FP32);

    C2PSANet gpu_net;
    gpu_net.nn.set_parameters(parameters);

    Loss gpu_loss(&gpu_net.nn, &cpu_net.dataset);
    gpu_loss.set_error(Loss::Error::MeanSquaredError);
    const VectorR gpu_gradient = calculate_gradient(gpu_loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

    ASSERT_EQ(cpu_gradient.size(), gpu_gradient.size());
    ASSERT_GT(cpu_gradient.size(), 0);

    // Components far below the largest one carry no signal worth holding the
    // two backends to; everything above that is compared on its own scale.
    const float largest = cpu_gradient.array().abs().maxCoeff();
    ASSERT_GT(largest, 0.0f);
    const float floor_value = 1e-3f * largest;

    float worst = 0.0f;
    Index worst_index = 0;

    for (Index i = 0; i < cpu_gradient.size(); ++i)
    {
        const float relative = abs(cpu_gradient(i) - gpu_gradient(i))
                             / max(floor_value, abs(cpu_gradient(i)));

        if (relative > worst) { worst = relative; worst_index = i; }
    }

    EXPECT_LT(worst, 5e-3f)
        << "Worst relative CPU vs GPU gradient difference " << worst
        << " at parameter " << worst_index
        << " (cpu " << cpu_gradient(worst_index)
        << ", gpu " << gpu_gradient(worst_index) << ")";
}
