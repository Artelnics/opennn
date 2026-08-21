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

    C2PSANet net;
    const vector<Index> training_idx = net.dataset.get_sample_indices("Training");
    const vector<Index> input_idx    = net.dataset.get_feature_indices("Input");

    Configuration::instance().set(Device::CPU, Type::FP32);
    {
        Batch batch(batch_size, &net.dataset, net.nn.get_config());
        batch.fill(training_idx, input_idx, {}, {});
        ForwardPropagation fp(batch_size, &net.nn);
        net.nn.forward_propagate(batch.get_inputs(), fp, false);
        TensorView out = fp.get_outputs();
        const Index n = out.size();
        vector<float> cpu_out(n);
        copy_n(out.as<float>(), n, cpu_out.data());

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Batch batch_gpu(batch_size, &net.dataset, net.nn.get_config());
        batch_gpu.fill(training_idx, input_idx, {}, {});
        ForwardPropagation fp_gpu(batch_size, &net.nn);
        net.nn.forward_propagate(batch_gpu.get_inputs(), fp_gpu, false);
        TensorView out_gpu = fp_gpu.get_outputs();

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

    Configuration::instance().set();
}
