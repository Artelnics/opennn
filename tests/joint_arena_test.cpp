//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J O I N T   A R E N A   T E S T
//
//   Pins the joint forward/delta memory arena. Both allocation paths produce
//   identical numbers, so nothing else in the suite notices if the joint plan
//   silently stops engaging -- only the footprint changes.

#include "pch.h"

#include "opennn/back_propagation.h"
#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/loss.h"
#include "opennn/neural_network.h"
#include "opennn/tabular_dataset.h"
#include "opennn/tensor_types.h"

using namespace opennn;

namespace
{

struct Model
{
    TabularDataset dataset;
    NeuralNetwork neural_network;
    unique_ptr<Loss> loss;

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

    for (const TensorView& delta : back_propagation.layer_output_deltas)
        if (!lies_within(delta, arena)) ++outside;

    for (const vector<TensorView>& slots : back_propagation.backward_slots)
        for (const TensorView& slot : slots)
            if (!lies_within(slot, arena)) ++outside;

    return outside;
}

}

// The joint plan must actually engage when a Loss is handed to ForwardPropagation,
// and must stay disengaged when one is not.
TEST(JointArenaTest, JointPlanEngagesOnlyWithALoss)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    Model model(batch_size, 32, 4);

    ForwardPropagation separate(batch_size, &model.neural_network,
                                ForwardPropagationMode::Training);
    EXPECT_FALSE(separate.joint_delta_plan.valid);

    ForwardPropagation joint(batch_size, &model.neural_network,
                             ForwardPropagationMode::Training, {}, false,
                             model.loss.get());

    ASSERT_TRUE(joint.joint_delta_plan.valid);
    EXPECT_GT(joint.joint_delta_plan.delta_bytes, 0);
    EXPECT_FALSE(joint.joint_delta_plan.layout.entries.empty());
    EXPECT_EQ(joint.joint_delta_plan.offsets.size(),
              joint.joint_delta_plan.layout.entries.size());
}

// With the joint plan active BackPropagation must own no delta memory of its own,
// and every delta view must land inside the forward arena.
TEST(JointArenaTest, BackPropagationBindsIntoTheForwardArena)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    Model model(batch_size, 32, 4);

    ForwardPropagation joint(batch_size, &model.neural_network,
                             ForwardPropagationMode::Training, {}, false,
                             model.loss.get());
    ASSERT_TRUE(joint.joint_delta_plan.valid);

    BackPropagation back_propagation(batch_size, model.loss.get(), &joint);

    EXPECT_EQ(back_propagation.delta_pool.bytes, 0)
        << "joint planning is active, so BackPropagation must not allocate a delta pool";

    ASSERT_GT(joint.data.bytes, 0);
    EXPECT_EQ(count_delta_views_outside(back_propagation, joint.data), 0)
        << "every delta view must point inside the forward arena";
}

// Without a Loss, BackPropagation falls back to owning its own pool. This is the
// other half of the branch in BackPropagation::set and must keep working.
TEST(JointArenaTest, SeparatePoolIsUsedWithoutTheJointPlan)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    Model model(batch_size, 32, 4);

    ForwardPropagation separate(batch_size, &model.neural_network,
                                ForwardPropagationMode::Training);
    ASSERT_FALSE(separate.joint_delta_plan.valid);

    BackPropagation back_propagation(batch_size, model.loss.get(), &separate);

    EXPECT_GT(back_propagation.delta_pool.bytes, 0)
        << "without a joint plan BackPropagation owns its delta pool";
    EXPECT_EQ(count_delta_views_outside(back_propagation,
                                        back_propagation.delta_pool), 0);
}

// A first-fit over the union of forward and delta lifetimes should never need more
// than the two independent peaks (you can always stack deltas above the forward
// block). Today it needs slightly MORE: exactly batch * outputs * sizeof(float),
// i.e. one output-delta tensor, measured constant across widths 64..1024 and
// batches 32..1024. Pin that gap so it cannot grow unnoticed; the target is zero.
TEST(JointArenaTest, JointArenaOverheadStaysBounded)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 64;
    const Index targets_number = 4;
    Model model(batch_size, 32, targets_number);

    ForwardPropagation separate(batch_size, &model.neural_network,
                                ForwardPropagationMode::Training);
    BackPropagation separate_back(batch_size, model.loss.get(), &separate);

    const Index separate_bytes = separate.data.bytes + separate_back.delta_pool.bytes;

    ForwardPropagation joint(batch_size, &model.neural_network,
                             ForwardPropagationMode::Training, {}, false,
                             model.loss.get());
    BackPropagation joint_back(batch_size, model.loss.get(), &joint);

    const Index joint_bytes = joint.data.bytes + joint_back.delta_pool.bytes;

    ASSERT_TRUE(joint.joint_delta_plan.valid);

    const Index known_gap = batch_size * targets_number * Index(sizeof(float));

    EXPECT_LE(joint_bytes, separate_bytes + known_gap)
        << "joint arena " << joint_bytes << " B vs separate " << separate_bytes
        << " B, over the known " << known_gap << " B gap";
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
