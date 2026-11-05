// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/random_utilities.h"
#include "opennn/core/json.h"
#include "opennn/dataset/dataset.h"
#include "opennn/core/device_backend.h"
#include "gtest/gtest.h"

#include <future>
#include <fstream>
#include <omp.h>

namespace opennn_test
{

inline void add_image_pattern(const std::filesystem::path& path)
{
    std::fstream image(path, std::ios::in | std::ios::out | std::ios::binary);
    image.seekg(54);
    std::vector<char> pixels((std::istreambuf_iterator<char>(image)), {});
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = char((i * 17 + i / 23) % 256);
    image.clear();
    image.seekp(54);
    image.write(pixels.data(), std::streamsize(pixels.size()));
}

inline void expect_cuda_augmentation(opennn::Dataset& dataset, float tolerance)
{
    using namespace opennn;
    auto indices = dataset.get_sample_indices(SampleRole::Training);
    ASSERT_FALSE(indices.empty());
    indices.resize(std::min<size_t>(indices.size(), 3));
    const auto features = dataset.get_feature_selection();
    for (Type type : {Type::FP32, Type::BF16})
    {
        if (type == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        Batch host(ssize(indices), &dataset, {Device::CPU, Type::FP32, 0});
        Batch staged(ssize(indices), &dataset, {Device::CUDA, type, 0}, BatchStorage::HostOnly);
        Batch destination(ssize(indices), &dataset, {Device::CUDA, type, 0}, BatchStorage::DeviceOnly);
        const auto read = [&](FillMode mode)
        {
            staged.fill(indices, features, mode);
            EXPECT_EQ(staged.image_transform.has_value(), mode == FillMode::Training);
            staged.upload_to_device_batch_async(destination, device::get_transfer_stream());
            // Untransformed uploads only record their completion on the transfer stream.
            staged.wait_h2d_on_compute_stream();
            std::vector<float> values(size_t(host.input.shape.size() + host.target.shape.size()));
            copy_device_to_host_float(destination.input.buffer.data(), destination.input.type,
                                      host.input.shape.size(), values.data(), device::get_compute_stream());
            device::copy_async(values.data() + host.input.shape.size(), destination.target.buffer.data(),
                               host.target.shape.size() * sizeof(float), device::CopyKind::DeviceToHost,
                               device::get_compute_stream());
            device::synchronize(device::get_compute_stream());
            return values;
        };
        for (Index epoch : {0, 1})
        {
            dataset.set_augmentation_epoch(epoch);
            std::reverse(indices.begin(), indices.end());
            for (FillMode mode : {FillMode::Training, FillMode::Validation, FillMode::Inference})
            {
                host.fill(indices, features, mode);
                const auto actual = read(mode);
                EXPECT_EQ(read(mode), actual);
                float maximum_error = 0.0f;
                for (Index i = 0; i < host.input.shape.size(); ++i)
                {
                    const float expected = destination.input.type == Type::BF16
                        ? float(Eigen::bfloat16(host.input.buffer.as<float>()[i])) : host.input.buffer.as<float>()[i];
                    maximum_error = std::max(maximum_error, std::abs(actual[size_t(i)] - expected));
                }
                EXPECT_LE(maximum_error, tolerance + (destination.input.type == Type::BF16 ? 0.004f : 0.0f));
                EXPECT_TRUE(std::equal(actual.begin() + host.input.shape.size(), actual.end(), host.target.buffer.as<float>()));
            }
        }
        const auto excluded = dataset.get_sample_indices(SampleRole::Validation);
        if (!excluded.empty())
        {
            auto invalid = indices;
            invalid.back() = excluded.front();
            EXPECT_THROW(staged.fill(invalid, features), std::runtime_error);
        }
    }
}

inline std::string augmentation_json(const opennn::Dataset& dataset)
{
    opennn::JsonWriter writer;
    dataset.to_JSON(writer);
    return writer.c_str();
}

inline void expect_augmentation_replay(opennn::Dataset& dataset, bool augmented_targets)
{
    using namespace opennn;
    const int previous_threads = omp_get_max_threads();
    ScopeExit restore_threads([&] { omp_set_num_threads(previous_threads); });
    omp_set_num_threads(1);
    const auto indices = dataset.get_sample_indices(SampleRole::Training);
    const auto features = dataset.get_feature_selection();
    const size_t input_size = size_t(dataset.get_input_shape().size());
    const size_t target_size = size_t(dataset.get_target_shape().size());
    const auto read = [&](const vector<Index>& samples)
    {
        pair<vector<float>, vector<float>> result{
            vector<float>(samples.size() * input_size), vector<float>(samples.size() * target_size)};
        dataset.fill_inputs(samples, features.inputs, result.first.data(), FillMode::Training);
        dataset.fill_targets(samples, features.targets, result.second.data(), FillMode::Training);
        return result;
    };

    dataset.set_augmentation_epoch(0);
    EXPECT_THROW(dataset.set_augmentation_epoch(-1), runtime_error);
    EXPECT_EQ(dataset.get_augmentation_epoch(), 0);
    set_seed(123);
    const float next_global_draw = random_uniform();
    set_seed(123);
    const auto expected = read(indices);
    EXPECT_EQ(random_uniform(), next_global_draw);

    vector<Index> reversed(indices.rbegin(), indices.rend());
    const vector<Index> subset{indices.back(), indices.front()};
    const auto start_reader = [&](vector<Index> samples, int threads)
    {
        return async(launch::async, [&, samples, threads]
        {
            omp_set_num_threads(threads);
            return read(samples);
        });
    };
    auto whole = start_reader(reversed, 4);
    auto part = start_reader(subset, 1);
    const auto check = [&](const vector<Index>& samples, const auto& actual)
    {
        for (size_t i = 0; i < samples.size(); ++i)
        {
            const size_t row = size_t(ranges::find(indices, samples[i]) - indices.begin());
            EXPECT_TRUE(equal(actual.first.begin() + i * input_size,
                              actual.first.begin() + (i + 1) * input_size,
                              expected.first.begin() + row * input_size));
            EXPECT_TRUE(equal(actual.second.begin() + i * target_size,
                              actual.second.begin() + (i + 1) * target_size,
                              expected.second.begin() + row * target_size));
        }
    };
    check(reversed, whole.get());
    check(subset, part.get());

    set_seed(456);
    EXPECT_NE(read(indices), expected);
    set_seed(123);
    dataset.set_augmentation_epoch(1);
    const auto next_epoch = read(indices);
    EXPECT_NE(next_epoch.first, expected.first);
    if (augmented_targets) EXPECT_NE(next_epoch.second, expected.second);
    dataset.set_augmentation_epoch(0);
    EXPECT_EQ(read(indices), expected);
}

}
