//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J O I N T   A R E N A   T E S T
//
//   Pins the joint forward/delta memory arena. Both allocation paths produce
//   identical numbers, so nothing else in the suite notices if the joint plan
//   silently stops engaging -- only the footprint changes.

#include "pch.h"

#include "opennn/neural_network/back_propagation.h"
#include "opennn/core/configuration.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/tensor_types.h"

using namespace opennn;

namespace
{

struct Model
{
    TabularDataset dataset;
    NeuralNetwork neural_network;
    unique_ptr<Loss> loss;

    // What Optimizer hands to ForwardPropagation: opaque lifetimes, no Loss.
    vector<MemoryPoolEntry> delta_lifetimes(Index batch) const
    {
        return BackPropagation::make_co_planned_lifetimes(neural_network, *loss, batch);
    }

    Model(Index samples_number, Index inputs_number, Index targets_number, Index width = 64)
        : dataset(samples_number, Shape{inputs_number}, Shape{targets_number})
    {
        dataset.set_data_random();
        dataset.set_sample_roles("Training");

        const Shape input_shape{inputs_number};

        neural_network.add_layer(
            make_unique<opennn::Dense>(input_shape, Shape{width}, "ReLU"), {-1});
        neural_network.add_layer(
            make_unique<opennn::Dense>(Shape{width}, Shape{width}, "ReLU"),
            {neural_network.get_layers_number() - 1});
        neural_network.add_layer(
            make_unique<opennn::Dense>(Shape{width}, Shape{targets_number}, "Identity"),
            {neural_network.get_layers_number() - 1});

        neural_network.compile();

        loss = make_unique<Loss>(&neural_network, &dataset);
        loss->set_error(Loss::Error::MeanSquaredError);
    }
};

// True when the view's whole byte range sits inside [base, base + bytes).
bool lies_within(const TensorView& view, const Buffer& arena)
{
    if (!view.data) return true;

    const auto* first = static_cast<const uint8_t*>(view.data);
    const auto* base = arena.as<uint8_t>();

    return first >= base && first + view.byte_size() <= base + arena.bytes;
}

Index count_delta_views_outside(const BackPropagation& back_propagation,
                                const Buffer& arena)
{
    Index outside = 0;

    for (const TensorView& delta : back_propagation.output_deltas)
        if (!lies_within(delta, arena)) ++outside;

    for (const vector<TensorView>& slots : back_propagation.slots)
        for (const TensorView& slot : slots)
            if (!lies_within(slot, arena)) ++outside;

    return outside;
}

}

// Co-planning must engage when lifetimes are supplied and stay disengaged otherwise.
TEST(JointArenaTest, CoPlanningEngagesOnlyWhenLifetimesAreSupplied)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    Model model(batch_size, 32, 4);

    ForwardPropagation separate(batch_size, &model.neural_network,
                                ForwardPropagationMode::Training);
    EXPECT_TRUE(separate.co_planned_offsets.empty());

    const vector<MemoryPoolEntry> lifetimes = model.delta_lifetimes(batch_size);
    ForwardPropagation joint(batch_size, &model.neural_network,
                             ForwardPropagationMode::Training, {}, false,
                             lifetimes);

    EXPECT_FALSE(lifetimes.empty());
    EXPECT_EQ(joint.co_planned_offsets.size(), lifetimes.size());
}

// With the joint plan active BackPropagation must own no delta memory of its own,
// and every delta view must land inside the forward arena.
TEST(JointArenaTest, BackPropagationBindsIntoTheForwardArena)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    Model model(batch_size, 32, 4);

    const vector<MemoryPoolEntry> lifetimes = model.delta_lifetimes(batch_size);
    ForwardPropagation joint(batch_size, &model.neural_network,
                             ForwardPropagationMode::Training, {}, false,
                             lifetimes);
    ASSERT_FALSE(joint.co_planned_offsets.empty());

    BackPropagation back_propagation(batch_size, model.loss.get(),
                                     &joint.arena, joint.co_planned_offsets);

    EXPECT_EQ(back_propagation.arena.bytes, 0)
        << "joint planning is active, so BackPropagation must not allocate an arena";

    ASSERT_GT(joint.arena.bytes, 0);
    EXPECT_EQ(count_delta_views_outside(back_propagation, joint.arena), 0)
        << "every delta view must point inside the forward arena";
}

// Without co-planned lifetimes, BackPropagation falls back to owning its own pool.
// This is the other half of the branch in BackPropagation::set and must keep working.
TEST(JointArenaTest, SeparatePoolIsUsedWithoutTheJointPlan)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    Model model(batch_size, 32, 4);

    ForwardPropagation separate(batch_size, &model.neural_network,
                                ForwardPropagationMode::Training);
    ASSERT_TRUE(separate.co_planned_offsets.empty());

    BackPropagation back_propagation(batch_size, model.loss.get(),
                                     &separate.arena, separate.co_planned_offsets);

    EXPECT_GT(back_propagation.arena.bytes, 0)
        << "without a joint plan BackPropagation owns its arena";
    EXPECT_EQ(count_delta_views_outside(back_propagation,
                                        back_propagation.arena), 0);
}

// A first-fit over the union should in principle never need more than the two
// independent peaks. With no early-release outputs it needs exactly
// batch * outputs * sizeof(float) more -- one output-delta tensor, constant across
// widths 64..1024 and batches 32..1024. That is the price of the Chronological
// ordering that activation recomputation depends on; see the comment at the
// plan_memory_pool call in ForwardPropagation::set. Architectures with early
// releases do not pay it: ResNet-50 measured -9.8% and a Transformer -6.3%
// against separate pools. Pin the gap so it cannot grow unnoticed.
TEST(JointArenaTest, JointArenaOverheadStaysBounded)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    const Index targets_number = 4;
    Model model(batch_size, 32, targets_number);

    ForwardPropagation separate(batch_size, &model.neural_network,
                                ForwardPropagationMode::Training);
    BackPropagation separate_back(batch_size, model.loss.get(),
                                  &separate.arena, separate.co_planned_offsets);

    const Index separate_bytes = separate.arena.bytes + separate_back.arena.bytes;

    const vector<MemoryPoolEntry> lifetimes = model.delta_lifetimes(batch_size);
    ForwardPropagation joint(batch_size, &model.neural_network,
                             ForwardPropagationMode::Training, {}, false,
                             lifetimes);
    BackPropagation joint_back(batch_size, model.loss.get(),
                               &joint.arena, joint.co_planned_offsets);

    const Index joint_bytes = joint.arena.bytes + joint_back.arena.bytes;

    ASSERT_FALSE(joint.co_planned_offsets.empty());

    const Index known_gap = batch_size * targets_number * Index(sizeof(float));

    EXPECT_LE(joint_bytes, separate_bytes + known_gap)
        << "joint arena " << joint_bytes << " B vs separate " << separate_bytes
        << " B, over the known " << known_gap << " B gap";
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
