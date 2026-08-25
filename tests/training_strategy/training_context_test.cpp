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
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/training_context.h"

using namespace opennn;

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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
