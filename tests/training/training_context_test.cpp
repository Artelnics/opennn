//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   C O N T E X T   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// TrainingContext exists to enforce three things its header spells out: the two
// halves are built in the one order that works, a context laid over another
// borrows its memory instead of allocating a second set, and a borrow that
// silently allocates anyway is refused rather than tolerated. That last one is
// the steady-state allocation guard, and none of the three had a test.

#include "tests/pch.h"

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training/loss.h"
#include "opennn/training/optimizer.h"
#include "opennn/training/training_context.h"

using namespace opennn;

namespace
{

void expect_optimizer_slot_layout_and_reset(Device allocation_device)
{
    const DeviceStream stream = allocation_device == Device::CUDA
        ? device::get_compute_stream() : nullptr;
    const vector<Shape> shapes = {
        {ALIGN_BYTES / Index(sizeof(float)) + 1}, {},
        {ALIGN_BYTES / Index(sizeof(opennn::bfloat16)) + 1}, {0},
        {ALIGN_BYTES + 1}, {1}
    };
    // The final slot deliberately has no explicit dtype and must stay FP32.
    const vector<Type> types = {Type::FP32, Type::BF16, Type::BF16, Type::INT8, Type::INT8};
    OptimizerData state;
    state.set(shapes, types, allocation_device);
    ASSERT_EQ(state.data.byte_size(), 7 * ALIGN_BYTES);
    ASSERT_EQ(state.views.size(), shapes.size());
    const auto* base = state.data.as<uint8_t>();
    for (size_t slot : {size_t(0), size_t(2), size_t(4), size_t(5)})
    {
        SCOPED_TRACE(slot);
        EXPECT_EQ(state.views[slot].get_shape(), shapes[slot]);
        EXPECT_EQ(state.views[slot].get_device(), allocation_device);
        EXPECT_TRUE(is_aligned(state.views[slot].get_data()));
    }
    EXPECT_EQ(state.views[0].get_data(), base);
    EXPECT_EQ(state.views[2].get_data(), base + 2 * ALIGN_BYTES);
    EXPECT_EQ(state.views[4].get_data(), base + 4 * ALIGN_BYTES);
    EXPECT_EQ(state.views[5].get_data(), base + 6 * ALIGN_BYTES);
    EXPECT_EQ(state.views[0].get_type(), Type::FP32);
    EXPECT_EQ(state.views[2].get_type(), Type::BF16);
    EXPECT_EQ(state.views[4].get_type(), Type::INT8);
    EXPECT_EQ(state.views[5].get_type(), Type::FP32);
    for (size_t slot : {size_t(1), size_t(3)})
    {
        EXPECT_TRUE(state.views[slot].empty());
        EXPECT_EQ(state.views[slot].get_data(), nullptr);
        EXPECT_EQ(state.views[slot].get_type(), Type::FP32);
        EXPECT_EQ(state.views[slot].get_device(), Device::CPU);
    }

    const auto expect_zero_bytes = [&]
    {
        vector<uint8_t> bytes(size_t(state.data.byte_size()), 0xff);
        device::copy_async(bytes.data(), state.data.data(), state.data.byte_size(),
                           allocation_device, Device::CPU, stream);
        if (allocation_device == Device::CUDA) device::synchronize(stream);
        EXPECT_TRUE(ranges::all_of(bytes, [](uint8_t value) { return value == 0; }));
    };
    expect_zero_bytes();
    const vector<uint8_t> dirty_bytes(size_t(state.data.byte_size()), 0xa5);
    device::copy_async(state.data.data(), dirty_bytes.data(), state.data.byte_size(),
                       Device::CPU, allocation_device, stream);
    // Reuse the allocation: reset must clear payload and alignment padding,
    // ordered after the preceding upload on the CUDA compute stream.
    state.set(shapes, types, allocation_device);
    expect_zero_bytes();

    state.set({Shape{}, Shape{1}}, allocation_device);
    ASSERT_EQ(state.views.size(), 2);
    EXPECT_TRUE(state.views[0].empty());
    EXPECT_EQ(state.views[0].get_data(), nullptr);
    EXPECT_EQ(state.views[1].get_data(), state.data.data());
    EXPECT_EQ(state.views[1].get_type(), Type::FP32);
    EXPECT_EQ(state.views[1].get_device(), allocation_device);
    expect_zero_bytes();
    state.set({}, allocation_device);
    EXPECT_TRUE(state.data.empty());
    EXPECT_TRUE(state.views.empty());
}

}

TEST(OptimizerDataTest, MixedPrecisionSlotsDefaultToFp32AndReset)
{
    expect_optimizer_slot_layout_and_reset(Device::CPU);
}

#ifdef OPENNN_HAS_CUDA
TEST(OptimizerDataTest, MixedPrecisionSlotsDefaultToFp32AndResetCuda)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    expect_optimizer_slot_layout_and_reset(Device::CUDA);
}
#endif

class TrainingContextTest : public ::testing::Test
{
protected:

    static constexpr Index samples_number = 16;

    unique_ptr<TabularDataset> dataset;
    unique_ptr<ApproximationNetwork> network;
    unique_ptr<Loss> loss;

    void SetUp() override
    {
        dataset = make_unique<TabularDataset>(samples_number, Shape{2}, Shape{1});
        dataset->set_data_random();
        dataset->set_sample_roles("Training");

        network = make_unique<ApproximationNetwork>(Shape{2}, Shape{6}, Shape{1});

        loss = make_unique<Loss>(network.get(), dataset.get());
        loss->set_error(Loss::Error::MeanSquaredError);
    }
};


TEST_F(TrainingContextTest, AStandaloneContextOwnsItsArena)
{
    const TrainingContext context(8, *loss);

    EXPECT_FALSE(context.shares_memory());
    EXPECT_EQ(context.forward.batch_size, 8);
    EXPECT_FALSE(context.backward.has_joint_gradient_arena());
    ASSERT_EQ(context.backward.get_gradient_slices().size(), 1);
    EXPECT_EQ(context.backward.get_gradient_slices()[0].values.get_data(),
              context.backward.gradient.data());
    EXPECT_EQ(context.backward.get_gradient_slices()[0].parameter_offset, 0);
}


TEST_F(TrainingContextTest, ASmallerContextBorrowsTheArenaOfAWholeBatch)
{
    TrainingContext whole_batch(8, *loss);

    // The remainder batch: smaller, and running only after every whole batch has
    // been consumed, so laying it over the same memory is safe.
    const TrainingContext remainder(3, *loss, false, &whole_batch);

    EXPECT_TRUE(remainder.shares_memory());
    EXPECT_FALSE(whole_batch.shares_memory());
    EXPECT_EQ(remainder.forward.batch_size, 3);
}


TEST_F(TrainingContextTest, ABorrowThatWouldNotFitIsRefused)
{
    TrainingContext small(2, *loss);

    // Asking a two-sample arena to host sixteen samples cannot work. The point
    // is that it throws rather than quietly allocating a second arena, which
    // would break the steady-state guarantee without anything noticing.
    EXPECT_THROW(TrainingContext(samples_number, *loss, false, &small), runtime_error);
}


TEST_F(TrainingContextTest, AnEqualSizedBorrowFits)
{
    TrainingContext first(8, *loss);

    // The guard is about fitting, not about being strictly smaller: an arena
    // planned for eight samples hosts another eight.
    const TrainingContext second(8, *loss, false, &first);

    EXPECT_TRUE(second.shares_memory());
}


TEST_F(TrainingContextTest, JointGradientSlicesLiveInsideTheTrainingArena)
{
    const TrainingContext context(8, *loss, false, nullptr,
                                  /*joint_gradient_arena*/ true);

    ASSERT_TRUE(context.backward.has_joint_gradient_arena());
    EXPECT_TRUE(context.backward.gradient.empty());
    EXPECT_EQ(context.backward.gradient_logical_bytes(),
              network->get_parameters_buffer_size() * Index(sizeof(float)));

    const auto* arena_begin =
        static_cast<const uint8_t*>(context.forward.arena.data());
    const auto* arena_end = arena_begin + context.forward.arena.byte_size();

    Index covered_elements = 0;
    for(const BackPropagation::GradientSlice& slice :
        context.backward.get_gradient_slices())
    {
        const auto* begin =
            static_cast<const uint8_t*>(slice.values.get_data());
        const auto* end = begin + slice.values.byte_size();
        EXPECT_GE(begin, arena_begin);
        EXPECT_LE(end, arena_end);
        covered_elements += slice.values.size();
    }

    EXPECT_EQ(covered_elements, network->get_parameters_buffer_size());
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
