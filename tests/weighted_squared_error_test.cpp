#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/loss.h"
#include "../opennn/standard_networks.h"

using namespace opennn;

TEST(WeightedSquaredErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_neural_network() == nullptr, true);
    EXPECT_EQ(loss.get_dataset() == nullptr, true);
}


TEST(WeightedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::WeightedSquaredError);

    EXPECT_EQ(loss.get_neural_network() != nullptr, true);
    EXPECT_EQ(loss.get_dataset() != nullptr, true);
}


TEST(WeightedSquaredErrorTest, BackPropagate)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);
    const Index outputs_number = 1;

    Dataset data_set(samples_number, { inputs_number }, { outputs_number });
    data_set.set_data_binary_classification();

    ClassificationNetwork neural_network({ inputs_number }, { neurons_number }, { outputs_number });

    Loss loss(&neural_network, &data_set);
    loss.set_error(Loss::Error::WeightedSquaredError);

    const VectorR analytical_gradient = loss.calculate_gradient();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    EXPECT_TRUE(are_equal(analytical_gradient, numerical_gradient, type(1.0e-2)));
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lewser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lewser General Public License for more details.

// You should have received a copy of the GNU Lewser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
