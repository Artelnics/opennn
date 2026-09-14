//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D E R I V A T I V E S   ( T E S T   H E L P E R )

#include "tests/numerical_derivatives.h"

#include "opennn/dataset/dataset.h"
#include "opennn/network/network.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/network/back_propagation.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/device_backend.h"

#include <Eigen/Dense>

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


float calculate_h(const float x)
{
    constexpr float finite_difference_step = 1e-3f;
    return finite_difference_step * (1.0f + abs(x));
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

    VectorR perturbed = parameters;
    VectorR numerical_gradient = VectorR::Zero(parameters_number);

    const auto error_at = [&]() -> float
    {
        setup.network->forward_propagate(setup.batch.get_inputs(),
                                                perturbed,
                                                setup.forward_propagation);
        return loss.calculate_error(setup.batch, setup.forward_propagation).error;
    };

    for (Index i = 0; i < parameters_number; ++i)
    {
        const float h = calculate_h(parameters(i));

        perturbed(i) += h;
        const float error_forward = error_at();

        perturbed(i) -= 2.0f * h;
        const float error_backward = error_at();

        perturbed(i) += h;

        numerical_gradient(i) = (error_forward - error_backward) / (2.0f * h);
    }

    return numerical_gradient;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
