// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#include "tests/common/pch.h"
#include "opennn/core/random_utilities.h"
#include "tests/common/numerical_derivatives.h"

#include "opennn/dataset/dataset.h"
#include "opennn/models/models.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training/loss.h"

using namespace opennn;

TEST(WeightedSquaredErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_network() == nullptr, true);
    EXPECT_EQ(loss.get_dataset() == nullptr, true);
}

TEST(WeightedSquaredErrorTest, GeneralConstructor)
{
    Network network;
    TabularDataset dataset;

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::WeightedSquaredError);

    EXPECT_EQ(loss.get_network() != nullptr, true);
    EXPECT_EQ(loss.get_dataset() != nullptr, true);
}

TEST(WeightedSquaredErrorTest, BackPropagate)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);
    const Index outputs_number = 1;

    TabularDataset data_set(samples_number, { inputs_number }, { outputs_number });
    data_set.set_data_binary_classification();

    ClassificationNetwork network({ inputs_number }, { neurons_number }, { outputs_number });

    Loss loss(&network, &data_set);
    loss.set_error(Loss::Error::WeightedSquaredError);

    const VectorR analytical_gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((analytical_gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-2));
}
