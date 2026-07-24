// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "pch.h"

#include "opennn/addition_layer.h"
#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/memory_pool.h"
#include "opennn/neural_network.h"

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
        {128, 0, 3},  // producer used by three later branches
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
    EXPECT_TRUE(overlaps(0, 4));  // safe reuse after the last fanout consumer
    EXPECT_GE(plan.peak_bytes, plan.lower_bound_live_bytes);
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

    // A detached/leaf output is externally observable after the full pass and
    // therefore must not be overwritten by later layers.
    const TensorView expected_leaf = training_layout.forward_slots[3].back();
    const TensorView actual_leaf = inference_layout.forward_slots[3].back();
    ASSERT_EQ(expected_leaf.size(), actual_leaf.size());
    for (Index i = 0; i < expected_leaf.size(); ++i)
        EXPECT_NEAR(expected_leaf.as<float>()[i],
                    actual_leaf.as<float>()[i],
                    1.0e-6f);

    EXPECT_LT(inference_layout.data.bytes, training_layout.data.bytes);

    // residual_add starts after stem's last direct branch has executed, so
    // first-fit can safely recycle that block.
    EXPECT_EQ(inference_layout.forward_slots[0].back().data,
              inference_layout.forward_slots[4].back().data);

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

    const auto& slots = inference_layout.forward_slots.front();
    ASSERT_FALSE(slots[1].empty());
    ASSERT_FALSE(slots[4].empty());
    ASSERT_FALSE(slots[5].empty());
    EXPECT_NE(slots[1].data, slots[4].data);
    EXPECT_NE(slots[1].data, slots[5].data);
    EXPECT_NE(slots[4].data, slots[5].data);

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
