// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/random_utilities.h"
#include "opennn/core/json.h"
#include "opennn/dataset/dataset.h"
#include "gtest/gtest.h"

#include <future>
#include <omp.h>

namespace opennn_test
{

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
