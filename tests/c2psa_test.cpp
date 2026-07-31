#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/c2psa_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/neural_network.h"
#include "opennn/tabular_dataset.h"
#include "opennn/loss.h"
#include "opennn/configuration.h"
#include "opennn/device_backend.h"

using namespace opennn;

// Tiny C2PSA: h=2, w=2, channels=8 → tokens=4, H=4, Attn[4×4].
// C2PSA parameters: 3×[4×4]=48 (Q/K/V) + [8×8]=64 (Wout) = 112.
// Small enough for a complete numerical gradient check in <5s.

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

// ---------------------------------------------------------------------------
// Test 1: CPU — analytical gradient matches finite differences.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test 2: GPU — same check on CUDA device.
// ---------------------------------------------------------------------------

TEST(C2PSA, GpuGradientMatchesNumerical)
{
    if (!opennn::device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);

    C2PSANet net;
    auto loss = net.make_loss();

    const VectorR analytical = calculate_gradient(*loss);
    const VectorR numerical  = calculate_numerical_gradient(*loss);

    Configuration::instance().set(Device::CPU, Type::FP32);

    ASSERT_EQ(analytical.size(), numerical.size());
    const float max_diff = (analytical - numerical).array().abs().maxCoeff();
    EXPECT_LT(max_diff, 1e-3f)
        << "Max element-wise diff (CUDA): " << max_diff;
}

// ---------------------------------------------------------------------------
// Test 3: CPU and GPU forward produce the same output (within float precision).
// ---------------------------------------------------------------------------

TEST(C2PSA, CpuAndGpuForwardOutputsMatch)
{
    if (!opennn::device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    const Index batch_size = SAMPLES;

    // Build net once; parameters are fixed for both forward passes.
    C2PSANet net;
    const vector<Index> training_idx = net.dataset.get_sample_indices("Training");
    const vector<Index> input_idx    = net.dataset.get_feature_indices("Input");

    // --- CPU forward ---
    Configuration::instance().set(Device::CPU, Type::FP32);
    {
        Batch batch(batch_size, &net.dataset, net.nn.get_config());
        batch.fill(training_idx, input_idx, {}, {});
        ForwardPropagation fp(batch_size, &net.nn);
        net.nn.forward_propagate(batch.get_inputs(), fp, false);
        TensorView out = fp.get_outputs();
        const Index n = out.size();
        vector<float> cpu_out(n);
        std::copy_n(out.as<float>(), n, cpu_out.data());

        // --- GPU forward ---
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
            max_diff = std::max(max_diff, std::abs(cpu_out[i] - gpu_out[i]));

        EXPECT_LT(max_diff, 1e-4f)
            << "Max CPU vs GPU forward output diff: " << max_diff;
    }
}
