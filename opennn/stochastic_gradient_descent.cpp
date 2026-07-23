//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "loss.h"
#include "profiler.h"
#include "batch.h"
#include "device_backend.h"
#include "stochastic_gradient_descent.h"
#include "kernel.cuh"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void update_parameters_cuda(NeuralNetwork* neural_network,
                                   BackPropagation& back_propagation,
                                   OptimizerData& optimizer_data,
                                   float current_learning_rate,
                                   float momentum,
                                   bool nesterov)
{
    const Index parameters_number = neural_network->get_parameters_size();

    float* const velocity_ptr = optimizer_data.views.empty()
        ? nullptr
        : optimizer_data.views[StochasticGradientDescent::Velocity].as<float>();

    PROFILE_SCOPE("optim:sgd_update_cuda");
    sgd_update_cuda(
        parameters_number,
        neural_network->get_parameters_data(),
        velocity_ptr,
        back_propagation.gradient.as<float>(),
        current_learning_rate,
        momentum,
        nesterov,
        neural_network->get_parameters_bf16_mirror_data());
}

#else

static void update_parameters_cuda(NeuralNetwork*,
                                   BackPropagation&,
                                   OptimizerData&,
                                   float,
                                   float,
                                   bool)
{
    throw runtime_error("update_parameters_cuda requires CUDA support.");
}

#endif

StochasticGradientDescent::StochasticGradientDescent(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void StochasticGradientDescent::set_default()
{
    name = "StochasticGradientDescent";


    initial_learning_rate = 0.001f;
    initial_decay = 0.001f;
    momentum = 0.0f;
    nesterov = false;
    batch_size = 0;


    training_loss_goal = 0.0f;
    maximum_time = 3600.0f;
    maximum_epochs = 1000;


    display_period = 100;
}

void StochasticGradientDescent::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}

void StochasticGradientDescent::set_initial_learning_rate(const float new_learning_rate)
{
    initial_learning_rate = new_learning_rate;
}

void StochasticGradientDescent::set_initial_decay(const float new_decay)
{
    initial_decay = new_decay;
}

void StochasticGradientDescent::set_momentum(const float new_momentum)
{
    momentum = new_momentum;
}

void StochasticGradientDescent::set_nesterov(bool new_nesterov_momentum)
{
    nesterov = new_nesterov_momentum;
}

void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  OptimizerData& optimizer_data)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimizer_data.iteration++;

    if (current_learning_rate == 0.0f)
        return;

    throw_if(momentum > 0.0f && optimizer_data.views.empty(),
             "StochasticGradientDescent::update_parameters: velocity buffer is not initialized.");

    clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);

    if (neural_network->is_gpu())
    {
        update_parameters_cuda(neural_network, back_propagation, optimizer_data,
                               current_learning_rate, momentum, nesterov);
        return;
    }

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    VectorMap gradient(back_propagation.gradient.as<float>(),
                       back_propagation.gradient.size_in_floats());

    const Index parameters_size = parameters.size();

    if (momentum <= 0.0f)
    {
        #pragma omp parallel for
        for (Index i = 0; i < parameters_size; ++i)
        {
            parameters(i) -= current_learning_rate * gradient(i);
        }
    }
    else
    {
        VectorMap velocity = optimizer_data.views[Velocity].as_vector();

        #pragma omp parallel for
        for (Index i = 0; i < parameters_size; ++i)
        {
            const float learning_rate_gradient = current_learning_rate * gradient(i);
            const float new_velocity = momentum * velocity(i) - learning_rate_gradient;
            velocity(i) = new_velocity;
            parameters(i) += nesterov ? momentum * new_velocity - learning_rate_gradient : new_velocity;
        }
    }
}

#ifdef OPENNN_HAS_CUDA
void StochasticGradientDescent::update_parameters_capturable(BackPropagation& back_propagation,
                                                             OptimizerData& optimizer_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);

    float* const velocity_ptr = optimizer_data.views.empty()
        ? nullptr
        : optimizer_data.views[Velocity].as<float>();

    sgd_update_capturable_cuda(
        neural_network->get_parameters_size(),
        neural_network->get_parameters_data(),
        velocity_ptr,
        back_propagation.gradient.as<float>(),
        optimizer_data.graph_effective_lr.as<float>(),
        momentum,
        nesterov,
        neural_network->get_parameters_bf16_mirror_data(),
        Backend::get_compute_stream());
}
#else
void StochasticGradientDescent::update_parameters_capturable(BackPropagation&, OptimizerData&) const
{
    throw runtime_error("update_parameters_capturable requires CUDA support.");
}
#endif

void StochasticGradientDescent::setup_optimizer_data(OptimizerData& optimizer_data,
                                                     Index parameters_number,
                                                     Device device,
                                                     [[maybe_unused]] bool on_gpu)
{
    if (momentum > 0.0f)
        optimizer_data.set({Shape{parameters_number}}, device);

    optimizer_data.iteration = 1;

    // The warmup runs before any on_epoch_begin, so it must see the epoch-0
    // learning rate.
    current_learning_rate = initial_learning_rate;

#ifdef OPENNN_HAS_CUDA
    if (on_gpu)
    {
        optimizer_data.graph_effective_lr.resize_bytes(Index(sizeof(float)), Device::CUDA);
        optimizer_data.graph_effective_lr.setZero();

        graph_update = [this, &optimizer_data](BackPropagation& back_propagation) {
            update_parameters_capturable(back_propagation, optimizer_data);
        };
    }
#endif
}

void StochasticGradientDescent::on_epoch_begin(Index epoch, [[maybe_unused]] OptimizerData& optimizer_data)
{
    current_learning_rate = initial_learning_rate / (1.0f + float(epoch) * initial_decay);

#ifdef OPENNN_HAS_CUDA
    if (graph_update)
        set_scalar_device_cuda(optimizer_data.graph_effective_lr.as<float>(),
                               current_learning_rate,
                               Backend::get_compute_stream());
#endif
}

void StochasticGradientDescent::to_JSON(JsonWriter& printer) const
{
    printer.open_element("StochasticGradientDescent");

    write_json(printer, {
        {"BatchSize", to_string(batch_size)},
        {"InitialLearningRate", to_string(initial_learning_rate)},
        {"InitialDecay", to_string(initial_decay)},
        {"Momentum", to_string(momentum)},
        {"Nesterov", to_string(nesterov)},
        {"ApplyMomentum", to_string(momentum > 0.0f)}
    });
    write_common_json(printer);

    printer.close_element();
}

void StochasticGradientDescent::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "StochasticGradientDescent");

    set_batch_size(read_json_index(root_element, "BatchSize"));

    if (root_element->has("InitialLearningRate")) set_initial_learning_rate(read_json_float(root_element, "InitialLearningRate"));
    if (root_element->has("InitialDecay"))        set_initial_decay(read_json_float(root_element, "InitialDecay"));
    if (root_element->has("Nesterov"))            set_nesterov(read_json_bool(root_element, "Nesterov"));

    if (root_element->has("Momentum"))
        set_momentum(read_json_float(root_element, "Momentum"));
    else
        set_momentum(read_json_bool(root_element, "ApplyMomentum") ? 0.9f : 0.0f);

    read_common_json(root_element);
}

REGISTER(Optimizer, StochasticGradientDescent, "StochasticGradientDescent");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
