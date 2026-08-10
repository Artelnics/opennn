// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "tests/pch.h"

#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/core/configuration.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/memory_pool.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/layers/scaling_layer.h"

using namespace opennn;

TEST(MemoryPoolTest, AllocatesStartsBeforeReleasingEnds)
{
    const vector<MemoryPoolEntry> entries = {
        {64, 0, 1},
        {64, 1, 2},
        {64, 2, 2}
    };

    const MemoryPoolPlan plan = plan_memory_pool(entries);

    ASSERT_EQ(plan.byte_offsets.size(), entries.size());
    EXPECT_EQ(plan.byte_offsets[0], 0);
    EXPECT_EQ(plan.byte_offsets[1], 64);
    EXPECT_EQ(plan.byte_offsets[2], 0);
    EXPECT_EQ(plan.peak_bytes, 128);
    EXPECT_EQ(plan.lower_bound_live_bytes, 128);
    EXPECT_EQ(plan.fragmentation_bytes(), 0);
}

TEST(MemoryPoolTest, KeepsFanoutProducerUntilLastConsumer)
{
    const vector<MemoryPoolEntry> entries = {
        {128, 0, 3},
        {64,  1, 4},
        {64,  2, 4},
        {64,  3, 4},
        {128, 4, 4}
    };

    const MemoryPoolPlan plan = plan_memory_pool(entries);

    const auto overlaps = [&](size_t a, size_t b)
    {
        const Index a_begin = plan.byte_offsets[a];
        const Index a_end = a_begin + entries[a].bytes;
        const Index b_begin = plan.byte_offsets[b];
        const Index b_end = b_begin + entries[b].bytes;
        return a_begin < b_end && b_begin < a_end;
    };

    EXPECT_FALSE(overlaps(0, 1));
    EXPECT_FALSE(overlaps(0, 2));
    EXPECT_FALSE(overlaps(0, 3));
    EXPECT_TRUE(overlaps(0, 4));
    EXPECT_GE(plan.peak_bytes, plan.lower_bound_live_bytes);
}

TEST(MemoryPoolTest, CompactLargestFirstEliminatesAvoidableFragmentation)
{
    const vector<MemoryPoolEntry> entries = {
        {56,  3, 3},
        {80,  4, 5},
        {80,  0, 6},
        {128, 0, 4},
        {112, 3, 5}
    };

    const MemoryPoolPlan plan =
        plan_memory_pool(entries, MemoryPoolStrategy::Compact);

    EXPECT_EQ(plan.lower_bound_live_bytes, 400);
    EXPECT_EQ(plan.peak_bytes, 400);
    EXPECT_EQ(plan.fragmentation_bytes(), 0);
}

TEST(BackPropagationMemoryTest, FanoutAccumulationReusesConsumerDelta)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    constexpr Index batch = 3;
    const Shape sequence_shape{2, 4};
    const Shape feature_shape{4};

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, feature_shape, "Identity",
                                                 false, "stem"),
                      {-1});
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, feature_shape, "Identity",
                                                 false, "branch_a"),
                      {0});
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, feature_shape, "Identity",
                                                 false, "branch_b"),
                      {0});
    network.add_layer(make_unique<Addition>(sequence_shape, "merge", 2), {1, 2});
    network.add_layer(make_unique<Flatten>(sequence_shape), {3});
    network.add_layer(make_unique<opennn::Dense>(Shape{8}, Shape{2}, "Identity",
                                                 false, "output"),
                      {4});
    network.compile();

    Loss loss(&network, nullptr);
    BackPropagation back_propagation(batch, loss);

    TensorView& branch_a_delta = back_propagation.slots[1][1];
    TensorView& branch_b_delta = back_propagation.slots[2][1];
    TensorView& stem_output_delta = back_propagation.output_deltas[0];

    ASSERT_FALSE(branch_a_delta.empty());
    ASSERT_FALSE(branch_b_delta.empty());
    ASSERT_NE(branch_a_delta.data, branch_b_delta.data);
    EXPECT_TRUE(stem_output_delta.data == branch_a_delta.data
                || stem_output_delta.data == branch_b_delta.data);

    branch_a_delta.as_vector().setConstant(1.0f);
    branch_b_delta.as_vector().setConstant(2.0f);
    back_propagation.accumulate_output_deltas(0);

    for (Index i = 0; i < stem_output_delta.size(); ++i)
        EXPECT_FLOAT_EQ(stem_output_delta.as<float>()[i], 3.0f);

    Configuration::instance().set();
}

TEST(ForwardPropagationMemoryTest, InferenceReusesResidualAndPassthroughOutputs)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    constexpr Index batch = 3;
    const Shape sequence_shape{2, 4};

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, Shape{4}, "Tanh",
                                                 false, "stem"),
                      {-1});
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, Shape{4}, "Tanh",
                                                 false, "branch_a"),
                      {0});
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, Shape{4}, "Tanh",
                                                 false, "branch_b"),
                      {0});
    network.add_layer(make_unique<opennn::Dense>(sequence_shape, Shape{4}, "Tanh",
                                                 false, "detached_leaf"),
                      {0});
    network.add_layer(make_unique<Addition>(sequence_shape, "residual_add", 2),
                      {1, 2});
    network.add_layer(make_unique<Flatten>(sequence_shape), {4});
    network.add_layer(make_unique<opennn::Dense>(Shape{8}, Shape{3}, "Identity",
                                                 false, "output"),
                      {5});
    network.compile();
    network.set_parameters_random();

    Tensor3 inputs(batch, 2, 4);
    inputs.setRandom();
    const vector<TensorView> input_views = {
        TensorView(inputs.data(), {batch, 2, 4})
    };

    ForwardPropagation training_layout(
        batch, &network, ForwardPropagationMode::Training);
    ForwardPropagation inference_layout(
        batch, &network, ForwardPropagationMode::Inference);

    network.forward_propagate(input_views, training_layout, false);
    network.forward_propagate(input_views, inference_layout, false);

    const TensorView expected = training_layout.get_outputs();
    const TensorView actual = inference_layout.get_outputs();
    ASSERT_EQ(expected.size(), actual.size());
    for (Index i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(expected.as<float>()[i], actual.as<float>()[i], 1.0e-6f);

    const TensorView expected_leaf = training_layout.slots[3].back();
    const TensorView actual_leaf = inference_layout.slots[3].back();
    ASSERT_EQ(expected_leaf.size(), actual_leaf.size());
    for (Index i = 0; i < expected_leaf.size(); ++i)
        EXPECT_NEAR(expected_leaf.as<float>()[i],
                    actual_leaf.as<float>()[i],
                    1.0e-6f);

    EXPECT_LT(inference_layout.arena.bytes, training_layout.arena.bytes);

    EXPECT_EQ(inference_layout.slots[0].back().data,
              inference_layout.slots[4].back().data);

    Configuration::instance().set();
}

TEST(ForwardPropagationMemoryTest, SameLayerAuxiliariesNeverAlias)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    NeuralNetwork network;
    auto gated = make_unique<opennn::Dense>(Shape{2, 4}, Shape{8}, "Identity",
                                            false, "gated");
    gated->set_gated(true);
    network.add_layer(move(gated), {-1});
    network.compile();

    ForwardPropagation inference_layout(
        2, &network, ForwardPropagationMode::Inference);

    const auto& slots = inference_layout.slots.front();
    ASSERT_FALSE(slots[1].empty());
    ASSERT_FALSE(slots[4].empty());
    ASSERT_FALSE(slots[5].empty());
    EXPECT_NE(slots[1].data, slots[4].data);
    EXPECT_NE(slots[1].data, slots[5].data);
    EXPECT_NE(slots[4].data, slots[5].data);

    Configuration::instance().set();
}

TEST(ForwardPropagationMemoryTest, TrainingRecomputeScratchUsesFutureActivations)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    constexpr Index batch = 2;
    const Shape feature_shape{4, 4, 4};

    NeuralNetwork network;
    network.add_layer(make_unique<Convolutional>(
                          Shape{4, 4, 2}, Shape{1, 1, 2, 4}, "Identity",
                          Shape{1, 1}, "Same", true, "conv_1"),
                      {-1});
    network.add_layer(make_unique<Convolutional>(
                          feature_shape, Shape{1, 1, 4, 4}, "Identity",
                          Shape{1, 1}, "Same", true, "conv_2"),
                      {0});
    network.add_layer(make_unique<Convolutional>(
                          feature_shape, Shape{1, 1, 4, 4}, "Identity",
                          Shape{1, 1}, "Same", false, "output"),
                      {1});
    network.compile();
    network.set_training_activation_recomputation(true);

    ForwardPropagation layout(batch, &network, ForwardPropagationMode::Training);

    const auto specs = network.get_forward_specs(batch);
    Index expected_persistent_bytes = 0;
    for (size_t layer = 0; layer < specs.size(); ++layer)
        for (size_t slot = 0; slot < specs[layer].size(); ++slot)
            if (slot != network.get_layers()[layer]->get_recomputable_forward_slot())
                expected_persistent_bytes += get_aligned_bytes(specs[layer][slot]);

    EXPECT_EQ(layout.arena.bytes, expected_persistent_bytes);

    EXPECT_EQ(layout.slots[0][1].data,
              layout.slots[1][2].data);
    EXPECT_EQ(layout.slots[1][1].data,
              layout.slots[2].back().data);

    Configuration::instance().set();
}

TEST(ForwardPropagationMemoryTest, TrainingDoesNotAllocateSkippedLeadingScaling)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    constexpr Index batch = 3;
    const Shape feature_shape{4};

    NeuralNetwork network;
    network.add_layer(make_unique<Scaling>(feature_shape), {-1});
    network.add_layer(make_unique<opennn::Dense>(
                          feature_shape, Shape{2}, "Identity",
                          false, "output"),
                      {0});
    network.compile();

    const Index scaling_bytes =
        get_aligned_bytes(network.get_forward_specs(batch).front());
    ASSERT_GT(scaling_bytes, 0);

    ForwardPropagation layout(
        batch, &network, ForwardPropagationMode::Training, {}, true);

    ASSERT_EQ(layout.slots[0].size(), 1);
    EXPECT_TRUE(layout.slots[0].back().empty());
    EXPECT_EQ(layout.arena.bytes,
              get_aligned_bytes(network.get_forward_specs(batch)[1]));

    MatrixR inputs = MatrixR::Random(batch, feature_shape.size());
    network.forward_propagate(
        {TensorView(inputs.data(), Shape{batch}.append(feature_shape))},
        layout,
        true);

    ASSERT_FALSE(layout.inputs[1].empty());
    EXPECT_EQ(layout.inputs[1][0].data, inputs.data());

    Configuration::instance().set();
}

TEST(ForwardPropagationMemoryTest, TrainingReusesProjectionResidualOutput)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    constexpr Index batch = 2;
    const Shape input_shape{4, 4, 2};
    const Shape stage_shape{4, 4, 8};

    NeuralNetwork network;
    network.add_layer(make_unique<Convolutional>(
                          input_shape, Shape{1, 1, 2, 4}, "ReLU",
                          Shape{1, 1}, "Same", true, "stem"),
                      {-1});
    network.add_layer(make_unique<Convolutional>(
                          Shape{4, 4, 4}, Shape{1, 1, 4, 8}, "ReLU",
                          Shape{1, 1}, "Same", true, "main"),
                      {0});
    network.add_layer(make_unique<Convolutional>(
                          Shape{4, 4, 4}, Shape{1, 1, 4, 8}, "Identity",
                          Shape{1, 1}, "Same", true, "projection"),
                      {0});

    auto residual = make_unique<Convolutional>(
        stage_shape, Shape{1, 1, 8, 8}, "ReLU",
        Shape{1, 1}, "Same", true, "residual");
    residual->set_residual(true);
    network.add_layer(move(residual), {1, 2});

    network.add_layer(make_unique<Convolutional>(
                          stage_shape, Shape{1, 1, 8, 8}, "ReLU",
                          Shape{1, 1}, "Same", true, "later"),
                      {3});
    network.compile();
    network.set_training_activation_recomputation(true);

    ForwardPropagation layout(
        batch, &network, ForwardPropagationMode::Training);

    const TensorView& projection_output = layout.slots[2].back();
    const TensorView& later_output = layout.slots[4].back();
    ASSERT_FALSE(projection_output.empty());
    ASSERT_FALSE(later_output.empty());
    EXPECT_EQ(projection_output.byte_size(), later_output.byte_size());
    EXPECT_EQ(projection_output.data, later_output.data);
    EXPECT_EQ(layout.inputs[3][1].data, projection_output.data);

    Configuration::instance().set();
}

TEST(ForwardPropagationMemoryTest, InferenceLayoutRejectsTraining)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{4}, Shape{2}, "Identity"),
                      {-1});
    network.compile();

    MatrixR inputs = MatrixR::Random(3, 4);
    const vector<TensorView> input_views = {
        TensorView(inputs.data(), {3, 4})
    };
    ForwardPropagation inference_layout(
        3, &network, ForwardPropagationMode::Inference);

    EXPECT_THROW(network.forward_propagate(input_views, inference_layout, true),
                 runtime_error);

    Configuration::instance().set();
}
