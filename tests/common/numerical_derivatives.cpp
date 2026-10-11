// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#include "tests/common/numerical_derivatives.h"

#include "opennn/dataset/dataset.h"
#include "opennn/network/network.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/network/back_propagation.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/device_backend.h"

#include <Eigen/Dense>

#include <cmath>

namespace opennn
{

namespace
{

Network* checked_network(Loss& loss, const char* caller)
{
    Network* network = loss.get_network();
    throw_if(!network, "{}: neural network is not set.", caller);
    return network;
}


Dataset* checked_dataset(Loss& loss, const char* caller)
{
    Dataset* dataset = loss.get_dataset();
    throw_if(!dataset, "{}: dataset is not set.", caller);
    return dataset;
}


// Everything the three helpers below set up before they differ: the training
// batch, filled and uploaded, and a forward/back propagation sized to match.
struct TrainingSetup
{
    TrainingSetup(Loss& loss, const char* caller)
        : network(checked_network(loss, caller)),
          dataset(checked_dataset(loss, caller)),
          samples_number(dataset->get_samples_number("Training")),
          batch(samples_number, dataset, network->get_config()),
          forward_propagation(samples_number, network),
          back_propagation(samples_number, loss)
    {
        batch.fill(dataset->get_sample_indices("Training"), dataset->get_feature_selection());

#ifdef OPENNN_HAS_CUDA
        if (network->is_gpu())
        {
            batch.upload_to_device_batch_async(batch, device::get_transfer_stream());
            batch.wait_h2d_complete();
        }
#endif
    }

    Network*     network;
    Dataset*           dataset;
    Index              samples_number;
    Batch              batch;
    ForwardPropagation forward_propagation;
    BackPropagation    back_propagation;
};

}


float central_difference_step(const float x)
{
    // 2^-10 is exactly representable and close to the point where truncation and
    // single-precision rounding errors of a central difference balance for these
    // checks; cbrt(epsilon), about 5e-3, was too coarse near activation kinks.
    // The step grows with |x| beyond 1.
    constexpr float base_step = 1.0f / 1024.0f;
    return base_step * max(1.0f, abs(x));
}


float calculate_numerical_error(Loss& loss)
{
    TrainingSetup setup(loss, "calculate_numerical_error");

    setup.network->forward_propagate(setup.batch.get_inputs(), setup.forward_propagation);

    return loss.calculate_error(setup.batch, setup.forward_propagation).error;
}


// Network::compile() zeroes the parameters, and only the StandardNetworks
// builders randomise them afterwards -- a network assembled by hand from
// add_layer() reaches here with every weight at zero unless the test says
// otherwise. That is not a harmless starting point for a gradient check: with
// zero weights the delta reaching every layer but the last is zero, so most of
// the gradient is identically zero on both sides of the comparison and the
// check passes whatever the backward pass does. Twenty tests were in that state
// -- one had 1 live gradient component out of 432 -- so this is a hard failure
// rather than a warning, to keep them from drifting back.
static void require_live_parameters(Network& network)
{
    network.copy_parameters_host();
    const VectorMap parameters = network.get_parameters_map();

    if (parameters.size() == 0) return;

    ASSERT_GT(parameters.array().abs().maxCoeff(), 0.0f)
        << "every parameter of this network is zero, which makes the gradient "
           "check vacuous: call set_parameters_random() after compile()";
}

VectorR calculate_gradient(Loss& loss)
{
    TrainingSetup setup(loss, "calculate_gradient");

    require_live_parameters(*setup.network);

    setup.network->forward_propagate(setup.batch.get_inputs(), setup.forward_propagation, ForwardPropagationMode::Training);

    loss.back_propagate(setup.batch, setup.forward_propagation, setup.back_propagation);

    setup.back_propagation.gradient.migrate_to(Device::CPU);

    return setup.back_propagation.gradient.as_vector();
}


VectorR calculate_numerical_gradient(Loss& loss)
{
    TrainingSetup setup(loss, "calculate_numerical_gradient");

    setup.network->copy_parameters_host();

    const VectorMap parameters = setup.network->get_parameters_map();
    const Index parameters_number = parameters.size();

    VectorR shifted = parameters;
    VectorR numerical_gradient(parameters_number);

    // Error with one parameter moved by `offset`; the parameter is restored exactly afterwards.
    const auto error_with_offset = [&](Index parameter, float offset) -> float
    {
        shifted(parameter) = parameters(parameter) + offset;
        setup.network->forward_propagate(setup.batch.get_inputs(), shifted, setup.forward_propagation);
        shifted(parameter) = parameters(parameter);
        return loss.calculate_error(setup.batch, setup.forward_propagation).error;
    };

    for (Index parameter = 0; parameter < parameters_number; ++parameter)
    {
        const float step = central_difference_step(parameters(parameter));
        numerical_gradient(parameter) = (error_with_offset(parameter, step)
                                         - error_with_offset(parameter, -step)) / (2.0f * step);
    }

    return numerical_gradient;
}

}
