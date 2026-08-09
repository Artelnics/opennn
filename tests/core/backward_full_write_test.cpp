//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K W A R D   F U L L   W R I T E   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// average_pooling_3d_backward and upsample_backward used to pre-zero their
// gradient before launching, so a kernel that skipped an element still came out
// zero and nobody noticed. Both now rely on the kernel writing every element
// itself, which is a promise nothing in the type system enforces.
//
// So these tests poison the gradient buffer before each call: any element the
// kernel fails to write comes back as the sentinel instead of silently reading
// as a plausible zero. They also check the GPU result against the CPU path,
// including the padded rows where the two implementations disagree about how
// the zero gets there - CPU skips those rows, GPU multiplies by a zero mask.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel.cuh"
#endif

using namespace opennn;

namespace
{

// Far from any gradient these tests produce, and negative so a missed element
// cannot be mistaken for a magnitude that merely looks large.
constexpr float poison = -123456.0f;

struct PoolingCase
{
    const char* name;
    Index batch_size, sequence_length, features;
    // Rows listed here are written as all-zeros, which is what the pooling code
    // treats as padding. A batch item with every row listed has a valid count of
    // zero - the branch that used to `continue` and leave the memset's zeros.
    vector<pair<Index, Index>> padded_steps;
};

const vector<PoolingCase> pooling_cases = {
    {"no_padding",        3, 4, 5, {}},
    {"some_padding",      3, 4, 5, {{0, 3}, {1, 2}, {1, 3}}},
    {"one_row_all_padded", 3, 4, 5, {{1, 0}, {1, 1}, {1, 2}, {1, 3}}},
    {"every_row_padded",  2, 3, 4, {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}},
};

bool is_padded(const PoolingCase& test_case, Index batch, Index step)
{
    for (const auto& [padded_batch, padded_step] : test_case.padded_steps)
        if (padded_batch == batch && padded_step == step) return true;
    return false;
}

vector<float> make_pooling_input(const PoolingCase& test_case)
{
    vector<float> input(size_t(test_case.batch_size * test_case.sequence_length * test_case.features));

    float value = 1.0f;
    for (Index b = 0; b < test_case.batch_size; ++b)
        for (Index s = 0; s < test_case.sequence_length; ++s)
        {
            const bool padded = is_padded(test_case, b, s);
            for (Index f = 0; f < test_case.features; ++f)
            {
                const size_t i = size_t((b * test_case.sequence_length + s) * test_case.features + f);
                input[i] = padded ? 0.0f : value;
                value += 0.25f;
            }
        }

    return input;
}

vector<float> make_pooling_output_delta(const PoolingCase& test_case)
{
    vector<float> delta(size_t(test_case.batch_size * test_case.features));
    for (size_t i = 0; i < delta.size(); ++i)
        delta[i] = 0.5f + 0.125f * float(i);
    return delta;
}

vector<float> pooling_backward_on_cpu(const PoolingCase& test_case)
{
    vector<float> input = make_pooling_input(test_case);
    vector<float> output_delta = make_pooling_output_delta(test_case);
    vector<float> input_delta(input.size(), poison);

    const Shape input_shape{test_case.batch_size, test_case.sequence_length, test_case.features};
    const Shape delta_shape{test_case.batch_size, test_case.features};

    const TensorView input_view(input.data(), input_shape);
    const TensorView output_delta_view(output_delta.data(), delta_shape);
    TensorView input_delta_view(input_delta.data(), input_shape);

    average_pooling_3d_backward(input_view, output_delta_view, input_delta_view);

    return input_delta;
}

#ifdef OPENNN_HAS_CUDA

// Owns one device allocation for the lifetime of a test body.
struct DeviceArray
{
    float* data = nullptr;
    Index bytes = 0;

    explicit DeviceArray(const vector<float>& host)
        : bytes(Index(host.size() * sizeof(float)))
    {
        data = static_cast<float*>(device::allocate(Device::CUDA, bytes));
        device::copy_async(data, host.data(), bytes, device::CopyKind::HostToDevice);
        device::synchronize();
    }

    vector<float> to_host() const
    {
        vector<float> host(size_t(bytes) / sizeof(float));
        device::synchronize();
        device::copy_async(host.data(), data, bytes, device::CopyKind::DeviceToHost);
        device::synchronize();
        return host;
    }

    ~DeviceArray() { device::deallocate(Device::CUDA, data, bytes); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

vector<float> pooling_backward_on_gpu(const PoolingCase& test_case)
{
    const vector<float> input = make_pooling_input(test_case);
    const vector<float> output_delta = make_pooling_output_delta(test_case);
    const vector<float> poisoned(input.size(), poison);

    DeviceArray input_device(input);
    DeviceArray output_delta_device(output_delta);
    DeviceArray input_delta_device(poisoned);

    const Shape input_shape{test_case.batch_size, test_case.sequence_length, test_case.features};
    const Shape delta_shape{test_case.batch_size, test_case.features};

    const TensorView input_view(input_device.data, input_shape, Type::FP32, Device::CUDA);
    const TensorView output_delta_view(output_delta_device.data, delta_shape, Type::FP32, Device::CUDA);
    TensorView input_delta_view(input_delta_device.data, input_shape, Type::FP32, Device::CUDA);

    average_pooling_3d_backward(input_view, output_delta_view, input_delta_view);
    device::synchronize();

    return input_delta_device.to_host();
}

#endif

}

TEST(BackwardFullWrite, AveragePoolingCpuWritesEveryElement)
{
    for (const PoolingCase& test_case : pooling_cases)
    {
        const vector<float> result = pooling_backward_on_cpu(test_case);

        for (size_t i = 0; i < result.size(); ++i)
            EXPECT_NE(result[i], poison)
                << "CPU average pooling backward left element " << i
                << " unwritten for case '" << test_case.name << "'";
    }
}

#ifdef OPENNN_HAS_CUDA

TEST(BackwardFullWrite, AveragePoolingGpuWritesEveryElement)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    for (const PoolingCase& test_case : pooling_cases)
    {
        const vector<float> result = pooling_backward_on_gpu(test_case);

        for (size_t i = 0; i < result.size(); ++i)
            EXPECT_NE(result[i], poison)
                << "GPU average pooling backward left element " << i
                << " unwritten for case '" << test_case.name << "'";
    }
}

TEST(BackwardFullWrite, AveragePoolingGpuMatchesCpu)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    for (const PoolingCase& test_case : pooling_cases)
    {
        const vector<float> host = pooling_backward_on_cpu(test_case);
        const vector<float> device_result = pooling_backward_on_gpu(test_case);

        ASSERT_EQ(host.size(), device_result.size());

        for (Index b = 0; b < test_case.batch_size; ++b)
            for (Index s = 0; s < test_case.sequence_length; ++s)
                for (Index f = 0; f < test_case.features; ++f)
                {
                    const size_t i = size_t((b * test_case.sequence_length + s) * test_case.features + f);
                    EXPECT_NEAR(host[i], device_result[i], 1e-5f)
                        << "case '" << test_case.name << "' diverges at batch " << b
                        << " step " << s << " feature " << f;
                }
    }
}

TEST(BackwardFullWrite, UpsampleGpuWritesEveryElementAndMatchesReference)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    const int batch = 2, in_h = 3, in_w = 4, channels = 3, scale = 2;
    const int out_h = in_h * scale, out_w = in_w * scale;
    const size_t input_size = size_t(batch) * in_h * in_w * channels;
    const size_t output_size = size_t(batch) * out_h * out_w * channels;

    vector<float> output_delta(output_size);
    for (size_t i = 0; i < output_size; ++i)
        output_delta[i] = 0.125f * float(i % 17) - 0.5f;

    // Each input pixel collects the scale x scale block of output pixels it fed.
    vector<float> expected(input_size, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int ih = 0; ih < in_h; ++ih)
            for (int iw = 0; iw < in_w; ++iw)
                for (int c = 0; c < channels; ++c)
                {
                    float accumulator = 0.0f;
                    for (int dh = 0; dh < scale; ++dh)
                        for (int dw = 0; dw < scale; ++dw)
                        {
                            const int oh = ih * scale + dh;
                            const int ow = iw * scale + dw;
                            accumulator += output_delta[size_t(((b * out_h + oh) * out_w + ow) * channels + c)];
                        }
                    expected[size_t(((b * in_h + ih) * in_w + iw) * channels + c)] = accumulator;
                }

    DeviceArray output_delta_device(output_delta);
    DeviceArray input_delta_device(vector<float>(input_size, poison));

    upsample_backward_cuda(batch, in_h, in_w, channels, scale,
                           output_delta_device.data, input_delta_device.data);
    device::synchronize();

    const vector<float> result = input_delta_device.to_host();

    ASSERT_EQ(result.size(), expected.size());

    for (size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_NE(result[i], poison) << "upsample backward left element " << i << " unwritten";
        EXPECT_NEAR(result[i], expected[i], 1e-5f) << "upsample backward diverges at element " << i;
    }
}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
