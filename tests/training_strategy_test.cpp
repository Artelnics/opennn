#include "pch.h"

#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/training_strategy.h"
#include "opennn/standard_networks.h"

using namespace opennn;

TEST(TrainingStrategy, DefaultConstructor)
{
    TrainingStrategy training_strategy;

    EXPECT_EQ(training_strategy.get_neural_network(), nullptr);
    EXPECT_EQ(training_strategy.get_dataset(), nullptr);
}

TEST(TrainingStrategy, GeneralConstructor)
{
    TabularDataset dataset(10, {2}, {1});
    dataset.set_data_random();

    ApproximationNetwork neural_network({2}, {3}, {1});

    TrainingStrategy training_strategy_1(&neural_network, &dataset);

    EXPECT_EQ(training_strategy_1.get_neural_network(), &neural_network);
    EXPECT_EQ(training_strategy_1.get_dataset(), &dataset);
}

TEST(TrainingStrategy, RebindsLossDependencies)
{
    TabularDataset first_dataset(10, {2}, {1});
    TabularDataset second_dataset(10, {2}, {1});
    ApproximationNetwork first_network({2}, {3}, {1});
    ApproximationNetwork second_network({2}, {3}, {1});

    TrainingStrategy training_strategy(&first_network, &first_dataset);
    training_strategy.set_neural_network(&second_network);
    training_strategy.set_dataset(&second_dataset);

    ASSERT_NE(training_strategy.get_loss(), nullptr);
    EXPECT_EQ(training_strategy.get_loss()->get_neural_network(), &second_network);
    EXPECT_EQ(training_strategy.get_loss()->get_dataset(), &second_dataset);

    training_strategy.set();
    EXPECT_EQ(training_strategy.get_loss(), nullptr);
    EXPECT_EQ(training_strategy.get_optimization_algorithm(), nullptr);
}

TEST(TrainingStrategy, InitializesWhenNetworkIsSetLater)
{
    TabularDataset dataset(10, {2}, {1});
    ApproximationNetwork neural_network({2}, {3}, {1});

    TrainingStrategy training_strategy;
    training_strategy.set_dataset(&dataset);
    training_strategy.set_neural_network(&neural_network);

    ASSERT_NE(training_strategy.get_loss(), nullptr);
    ASSERT_NE(training_strategy.get_optimization_algorithm(), nullptr);
    EXPECT_EQ(training_strategy.get_loss()->get_neural_network(), &neural_network);
    EXPECT_EQ(training_strategy.get_loss()->get_dataset(), &dataset);
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
