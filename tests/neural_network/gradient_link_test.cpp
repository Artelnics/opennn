//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   L I N K   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A layer writes its parameter gradients through views, so only one
// BackPropagation's buffer is reachable at a time. Training with a remainder
// batch keeps two contexts alive and alternates between them every epoch, and
// the optimizer used to re-run BackPropagation::set on each switch just to move
// those views. The link now belongs to the backward pass: whichever context is
// handed to Loss::back_propagate is the one written.
//
// Without that, the pass below writes into whichever BackPropagation happened to
// be constructed last, and the caller's own buffer comes back untouched.

#include "tests/pch.h"

#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"

#include "gtest/gtest.h"

using namespace opennn;

namespace
{

float gradient_magnitude(BackPropagation& back_propagation)
{
    back_propagation.gradient.migrate_to(Device::CPU);

    const Index size = back_propagation.gradient.size_in_floats();
    const float* values = back_propagation.gradient.as<float>();

    float total = 0.0f;
    for (Index i = 0; i < size; ++i)
        total += abs(values[i]);

    return total;
}

}

TEST(GradientLink, BackPropagateWritesTheContextItIsGiven)
{
    set_seed(7u);

    TabularDataset dataset(4, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({2}, {3}, {1});
    network.set_parameters_glorot();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    // Through the base class: TabularDataset overrides the argument-less
    // get_samples_number, which hides the role-taking overloads.
    Dataset& base_dataset = dataset;

    const Index samples_number = base_dataset.get_samples_number("Training");

    Batch batch(samples_number, &dataset, network.get_config());
    batch.fill(base_dataset.get_sample_indices("Training"),
               base_dataset.get_feature_selection());

    ForwardPropagation forward_propagation(samples_number, &network);

    // Two contexts over one network, as a remainder-batch run keeps. `second` is
    // built last, so it is the one a construction-time link would leave wired up.
    BackPropagation first(samples_number, loss);
    BackPropagation second(samples_number, loss);

    ASSERT_EQ(gradient_magnitude(first), 0.0f);
    ASSERT_EQ(gradient_magnitude(second), 0.0f);

    network.forward_propagate(batch.get_inputs(), forward_propagation, true);
    loss.back_propagate(batch, forward_propagation, first);

    EXPECT_GT(gradient_magnitude(first), 0.0f)
        << "back_propagate wrote somewhere other than the context it was given";
    EXPECT_EQ(gradient_magnitude(second), 0.0f)
        << "back_propagate wrote into a context it was not given";

    // And back again, which is the per-epoch switch the tail batch performs.
    network.forward_propagate(batch.get_inputs(), forward_propagation, true);
    loss.back_propagate(batch, forward_propagation, second);

    EXPECT_GT(gradient_magnitude(second), 0.0f);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
