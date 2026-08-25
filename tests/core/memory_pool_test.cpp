// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "tests/pch.h"

#include <utility>

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

TEST(MemoryPoolTest, BothStrategiesRespectRecordedLifetimes)
{
    const vector<MemoryPoolEntry> entries = {
        {56,  3, 3},
        {80,  4, 5},
        {80,  0, 6},
        {128, 0, 4},
        {112, 3, 5}
    };

    for (const MemoryPoolStrategy strategy : {
             MemoryPoolStrategy::Chronological,
             MemoryPoolStrategy::Compact})
    {
        const MemoryPoolPlan plan = plan_memory_pool(entries, strategy);

        for (size_t i = 0; i < entries.size(); ++i)
            for (size_t j = i + 1; j < entries.size(); ++j)
            {
                const bool lifetimes_overlap =
                    entries[i].first_step <= entries[j].last_step
                    && entries[j].first_step <= entries[i].last_step;
                if (!lifetimes_overlap) continue;

                const bool memory_overlaps =
                    plan.byte_offsets[i] < plan.byte_offsets[j] + entries[j].bytes
                    && plan.byte_offsets[j] < plan.byte_offsets[i] + entries[i].bytes;
                EXPECT_FALSE(memory_overlaps);
            }
    }
}

TEST(BackPropagationMemoryTest, FanoutAccumulationReusesConsumerDelta)
{
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
    const vector<MemoryPoolEntry> lifetimes =
        BackPropagation::make_co_planned_lifetimes(loss, batch);
    const MemoryPoolPlan chronological_plan = plan_memory_pool(
        lifetimes, MemoryPoolStrategy::Chronological);
    const MemoryPoolPlan compact_plan = plan_memory_pool(
        lifetimes, MemoryPoolStrategy::Compact);
    ASSERT_LT(compact_plan.peak_bytes, chronological_plan.peak_bytes);

    BackPropagation back_propagation(batch, loss);

    EXPECT_EQ(back_propagation.arena.byte_size(), compact_plan.peak_bytes);

    TensorView& branch_a_delta = back_propagation.slots[1][1];
    TensorView& branch_b_delta = back_propagation.slots[2][1];
    TensorView& stem_output_delta = back_propagation.output_deltas[0];

    ASSERT_FALSE(branch_a_delta.empty());
    ASSERT_FALSE(branch_b_delta.empty());
    ASSERT_NE(branch_a_delta.get_data(), branch_b_delta.get_data());

    // branch_a (the first consumer) holds the stem's delta; branch_b's delta is
    // handed to branch_a's backward as an addend, summed by its input-delta
    // GEMM (Dense::folds_input_delta_addend), so the accumulation pass has
    // nothing left to add for this edge.
    ASSERT_EQ(stem_output_delta.get_data(), branch_a_delta.get_data());
    EXPECT_EQ(back_propagation.input_delta_addend(1, 0).get_data(), branch_b_delta.get_data());

    branch_a_delta.as_vector().setConstant(1.0f);
    branch_b_delta.as_vector().setConstant(2.0f);
    back_propagation.accumulate_output_deltas(0);

    for (Index i = 0; i < stem_output_delta.size(); ++i)
        EXPECT_FLOAT_EQ(stem_output_delta.as<float>()[i], 1.0f);
}

TEST(ForwardPropagationMemoryTest, InferenceReusesResidualAndPassthroughOutputs)
{
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

    network.forward_propagate(input_views, training_layout, ForwardPropagationMode::Inference);
    network.forward_propagate(input_views, inference_layout, ForwardPropagationMode::Inference);

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

    EXPECT_LT(inference_layout.arena.byte_size(), training_layout.arena.byte_size());

    EXPECT_EQ(inference_layout.slots[0].back().get_data(),
              inference_layout.slots[4].back().get_data());
}

TEST(ForwardPropagationMemoryTest, SameLayerAuxiliariesNeverAlias)
{
    NeuralNetwork network;
    auto gated = make_unique<opennn::Dense>(Shape{2, 4}, Shape{8}, "Identity",
                                            false, "gated");
    gated->set_gated(true);
    network.add_layer(std::move(gated), {-1});
    network.compile();

    ForwardPropagation inference_layout(
        2, &network, ForwardPropagationMode::Inference);

    const auto& slots = inference_layout.slots.front();
    const TensorView& combination = slots[1];
    const TensorView& activation = slots[4];
    const TensorView& output = slots.back();
    ASSERT_FALSE(combination.empty());
    ASSERT_FALSE(activation.empty());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(combination.get_data(), activation.get_data());
    EXPECT_NE(combination.get_data(), output.get_data());
    EXPECT_NE(activation.get_data(), output.get_data());
}

TEST(ForwardPropagationMemoryTest, TrainingRecomputeScratchUsesFutureActivations)
{
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
    // get_recomputable_forward_slot answers in slot ids, and spec i is slot
    // i + 1 -- slot 0 is the layer input, which is not allocated from a spec.
    for (size_t layer = 0; layer < specs.size(); ++layer)
        for (size_t spec = 0; spec < specs[layer].size(); ++spec)
            if (spec + 1 != network.get_layers()[layer]->get_recomputable_forward_slot())
                expected_persistent_bytes += get_aligned_bytes(specs[layer][spec]);

    EXPECT_EQ(layout.arena.byte_size(), expected_persistent_bytes);

    EXPECT_EQ(layout.slots[0][1].get_data(),
              layout.slots[1][2].get_data());
    EXPECT_EQ(layout.slots[1][1].get_data(),
              layout.slots[2].back().get_data());
}

TEST(ForwardPropagationMemoryTest, RecomputeOverlayUsesLifetimesAcrossLayerTypes)
{
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
                          Shape{1, 1}, "Same", false, "conv_3"),
                      {1});
    network.add_layer(make_unique<Addition>(feature_shape, "addition", 2),
                      {2, 2});
    network.compile();
    network.set_training_activation_recomputation(true);

    ForwardPropagation layout(batch, &network, ForwardPropagationMode::Training);

    EXPECT_EQ(layout.slots[0][1].get_data(), layout.slots[1][2].get_data());
    EXPECT_EQ(layout.slots[1][1].get_data(), layout.slots[2].back().get_data());
}

TEST(ForwardPropagationMemoryTest, TrainingDoesNotAllocateSkippedLeadingScaling)
{
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
    EXPECT_EQ(layout.arena.byte_size(),
              get_aligned_bytes(network.get_forward_specs(batch)[1]));

    MatrixR inputs = MatrixR::Random(batch, feature_shape.size());
    network.forward_propagate(
        {TensorView(inputs.data(), Shape{batch}.append(feature_shape))},
        layout,
        ForwardPropagationMode::Training);

    ASSERT_FALSE(layout.inputs[1].empty());
    EXPECT_EQ(layout.inputs[1][0].get_data(), inputs.data());
}

TEST(ForwardPropagationMemoryTest, TrainingReusesProjectionResidualOutput)
{
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
    network.add_layer(std::move(residual), {1, 2});

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
    EXPECT_EQ(projection_output.get_data(), later_output.get_data());
    EXPECT_EQ(layout.inputs[3][1].get_data(), projection_output.get_data());
}

TEST(ForwardPropagationMemoryTest, InferenceLayoutRejectsTraining)
{
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

    EXPECT_THROW(network.forward_propagate(input_views, inference_layout, ForwardPropagationMode::Training),
                 runtime_error);
}
