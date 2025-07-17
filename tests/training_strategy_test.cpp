#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"

using namespace opennn;

TEST(TrainingStrategy, DefaultConstructor)
{
    TrainingStrategy training_strategy;

    EXPECT_EQ(training_strategy.has_neural_network(), false);
    EXPECT_EQ(training_strategy.has_dataset(), false);
}


TEST(TrainingStrategy, GeneralConstructor)
{
    Dataset dataset;
    NeuralNetwork neural_network;

    TrainingStrategy training_strategy_1(&neural_network, &dataset);

    EXPECT_EQ(training_strategy_1.get_neural_network(), &neural_network);
    EXPECT_EQ(training_strategy_1.get_dataset(), &dataset);
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
