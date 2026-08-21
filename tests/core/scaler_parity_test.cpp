//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L E R   P A R I T Y   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Scaling is the only preprocessing stage with two independent implementations:
// scale_cpu in tensor_operations.cpp and the scale kernel in kernel_layers.cu.
// Nothing forces them to agree, so this file walks every scaler in both
// directions - including features with no spread, where the guards live - and
// asserts the two paths produce the same numbers.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/variable.h"
#include "opennn/neural_network/layers/scaling_layer.h"

using namespace opennn;

namespace
{

// One feature per interesting case. Healthy and degenerate variants of each
// scaler sit side by side so a guard that fires on only one path shows up as a
// single mismatched column rather than a whole-tensor difference.
struct Feature
{
    const char* name;
    ScalerMethod scaler;
    float minimum, maximum, mean, standard_deviation;
};

const vector<Feature> features_under_test = {
    {"none",              ScalerMethod::None,                 -3.0f,  4.0f,  0.5f, 1.5f},
    {"minmax",            ScalerMethod::MinimumMaximum,       -2.0f,  6.0f,  1.0f, 2.0f},
    {"minmax_flat",       ScalerMethod::MinimumMaximum,        3.0f,  3.0f,  3.0f, 0.0f},
    {"minmax_negative",   ScalerMethod::MinimumMaximum,      -10.0f, -4.0f, -7.0f, 1.5f},
    {"meanstd",           ScalerMethod::MeanStandardDeviation, 0.0f,  9.0f,  1.0f, 2.0f},
    {"meanstd_flat",      ScalerMethod::MeanStandardDeviation, 5.0f,  5.0f,  5.0f, 0.0f},
    {"std",               ScalerMethod::StandardDeviation,    -4.0f,  4.0f,  0.0f, 2.0f},
    {"std_flat",          ScalerMethod::StandardDeviation,     7.0f,  7.0f,  7.0f, 0.0f},
    {"logarithm",         ScalerMethod::Logarithm,             0.1f,  8.0f,  2.0f, 1.0f},
    {"image",             ScalerMethod::ImageMinMax,           0.0f,255.0f,128.0f,60.0f},
};

// Deliberately spans the awkward inputs: zero, negatives, and a value below the
// Logarithm clamp.
const vector<float> row_values = {0.0f, 1.0f, -1.0f, 2.5f, -7.0f, 255.0f, 1e-9f, 100.0f};

struct ScalerInputs
{
    Index rows = 0, features = 0;
    vector<float> input, minimums, maximums, means, standard_deviations, scalers;

    ScalerInputs()
        : rows(Index(row_values.size())), features(Index(features_under_test.size()))
    {
        input.resize(size_t(rows * features));
        for (Index r = 0; r < rows; ++r)
            for (Index f = 0; f < features; ++f)
                input[size_t(r * features + f)] = row_values[size_t(r)];

        for (const Feature& feature : features_under_test)
        {
            minimums.push_back(feature.minimum);
            maximums.push_back(feature.maximum);
            means.push_back(feature.mean);
            standard_deviations.push_back(feature.standard_deviation);
            scalers.push_back(float(int(feature.scaler)));
        }
    }
};

// A TensorView promises its data is ALIGN_BYTES-aligned - the accessors build
// Eigen maps with AlignedMax, and Eigen then emits aligned vector loads and
// stores. std::vector does not promise that much, so host storage for a view
// goes through Buffer, which allocates the alignment the views assume.
struct AlignedFloats
{
    explicit AlignedFloats(const vector<float>& values)
    {
        buffer.resize_bytes(Index(values.size() * sizeof(float)), Device::CPU);
        copy(values.begin(), values.end(), buffer.as<float>());
    }

    explicit AlignedFloats(size_t count)
    {
        buffer.resize_bytes(Index(count * sizeof(float)), Device::CPU);
        buffer.setZero();
    }

    float* data() { return buffer.as<float>(); }

    vector<float> to_vector(size_t count) const
    {
        return vector<float>(buffer.as<float>(), buffer.as<float>() + count);
    }

    Buffer buffer;
};

// Runs scale() or unscale() on the host and returns the result.
vector<float> run_on_cpu(const ScalerInputs& in, bool inverse, float min_range, float max_range)
{
    ScalerInputs data = in;

    AlignedFloats input_data(data.input);
    AlignedFloats minimums_data(data.minimums);
    AlignedFloats maximums_data(data.maximums);
    AlignedFloats means_data(data.means);
    AlignedFloats deviations_data(data.standard_deviations);
    AlignedFloats scalers_data(data.scalers);
    AlignedFloats output_data(data.input.size());

    const Shape matrix_shape{data.rows, data.features};
    const Shape vector_shape{data.features};

    const TensorView input_view(input_data.data(), matrix_shape);
    const TensorView minimums_view(minimums_data.data(), vector_shape);
    const TensorView maximums_view(maximums_data.data(), vector_shape);
    const TensorView means_view(means_data.data(), vector_shape);
    const TensorView deviations_view(deviations_data.data(), vector_shape);
    const TensorView scalers_view(scalers_data.data(), vector_shape);
    TensorView output_view(output_data.data(), matrix_shape);

    if (inverse)
        unscale(input_view, minimums_view, maximums_view, means_view, deviations_view,
                scalers_view, min_range, max_range, output_view);
    else
        scale(input_view, minimums_view, maximums_view, means_view, deviations_view,
              scalers_view, min_range, max_range, output_view);

    return output_data.to_vector(data.input.size());
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
    }

    explicit DeviceArray(Index float_count)
        : bytes(float_count * Index(sizeof(float)))
    {
        data = static_cast<float*>(device::allocate(Device::CUDA, bytes));
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

vector<float> run_on_gpu(const ScalerInputs& in, bool inverse, float min_range, float max_range)
{
    DeviceArray input(in.input);
    DeviceArray minimums(in.minimums);
    DeviceArray maximums(in.maximums);
    DeviceArray means(in.means);
    DeviceArray deviations(in.standard_deviations);
    DeviceArray scalers(in.scalers);
    DeviceArray output(Index(in.input.size()));

    const Shape matrix_shape{in.rows, in.features};
    const Shape vector_shape{in.features};

    const TensorView input_view(input.data, matrix_shape, Type::FP32, Device::CUDA);
    const TensorView minimums_view(minimums.data, vector_shape, Type::FP32, Device::CUDA);
    const TensorView maximums_view(maximums.data, vector_shape, Type::FP32, Device::CUDA);
    const TensorView means_view(means.data, vector_shape, Type::FP32, Device::CUDA);
    const TensorView deviations_view(deviations.data, vector_shape, Type::FP32, Device::CUDA);
    const TensorView scalers_view(scalers.data, vector_shape, Type::FP32, Device::CUDA);
    TensorView output_view(output.data, matrix_shape, Type::FP32, Device::CUDA);

    if (inverse)
        unscale(input_view, minimums_view, maximums_view, means_view, deviations_view,
                scalers_view, min_range, max_range, output_view);
    else
        scale(input_view, minimums_view, maximums_view, means_view, deviations_view,
              scalers_view, min_range, max_range, output_view);

    return output.to_host();
}

// Reports the first mismatch by feature name, so a failure says which scaler
// and which direction rather than just which flat index.
void expect_paths_agree(bool inverse, float min_range, float max_range)
{
    const ScalerInputs in;
    const vector<float> host = run_on_cpu(in, inverse, min_range, max_range);
    const vector<float> device_result = run_on_gpu(in, inverse, min_range, max_range);

    ASSERT_EQ(host.size(), device_result.size());

    for (Index r = 0; r < in.rows; ++r)
        for (Index f = 0; f < in.features; ++f)
        {
            const size_t i = size_t(r * in.features + f);
            EXPECT_NEAR(host[i], device_result[i], 1e-5f)
                << (inverse ? "unscale" : "scale") << " diverges on feature '"
                << features_under_test[size_t(f)].name << "' for input "
                << row_values[size_t(r)];
        }
}

#endif

}

#ifdef OPENNN_HAS_CUDA

TEST(ScalerParity, ScaleMatchesBetweenCpuAndGpu)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    expect_paths_agree(false, -1.0f, 1.0f);
}

TEST(ScalerParity, UnscaleMatchesBetweenCpuAndGpu)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    expect_paths_agree(true, -1.0f, 1.0f);
}

// The scalers are also used with an asymmetric target range, which changes
// which side of the guards the arithmetic lands on.
TEST(ScalerParity, MatchesBetweenCpuAndGpuOnAsymmetricRange)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";

    expect_paths_agree(false, 0.0f, 1.0f);
    expect_paths_agree(true, 0.0f, 1.0f);
}

#endif

// Runs in every build: a round trip through scale then unscale must return the
// original values for the features that carry enough information to invert.
TEST(ScalerParity, CpuRoundTripRecoversInvertibleFeatures)
{
    const ScalerInputs in;
    const vector<float> scaled = run_on_cpu(in, false, -1.0f, 1.0f);

    ScalerInputs round_trip = in;
    round_trip.input = scaled;
    const vector<float> recovered = run_on_cpu(round_trip, true, -1.0f, 1.0f);

    for (Index r = 0; r < in.rows; ++r)
        for (Index f = 0; f < in.features; ++f)
        {
            const Feature& feature = features_under_test[size_t(f)];

            // Logarithm clamps at EPSILON, so inputs at or below it are lost.
            if (feature.scaler == ScalerMethod::Logarithm && row_values[size_t(r)] <= EPSILON)
                continue;

            const bool has_spread = feature.scaler == ScalerMethod::MinimumMaximum
                ? feature.maximum - feature.minimum >= EPSILON
                : feature.standard_deviation >= EPSILON;

            // A feature with no spread held exactly one value, so that value -
            // not the input - is what a round trip must produce.
            const bool degenerate = !has_spread
                && (feature.scaler == ScalerMethod::MinimumMaximum
                    || feature.scaler == ScalerMethod::MeanStandardDeviation
                    || feature.scaler == ScalerMethod::StandardDeviation);

            const float original = row_values[size_t(r)];
            const float expected = degenerate
                ? (feature.scaler == ScalerMethod::MinimumMaximum ? feature.minimum : feature.mean)
                : original;

            const size_t i = size_t(r * in.features + f);
            EXPECT_NEAR(expected, recovered[i], 1e-3f * max(1.0f, abs(expected)))
                << "round trip lost feature '" << feature.name << "'";
        }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
