//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/stochastic_gradient_descent.h"

#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/error_functions.h"
#include "opennn/training_strategy/kernel_optimizers.cuh"
#include "opennn/training_strategy/loss.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void update_parameters_cuda(BackPropagation& back_propagation,
                                   OptimizerData& optimizer_data,
                                   float current_learning_rate,
                                   float momentum,
                                   bool nesterov)
{
    NeuralNetwork* const neural_network = back_propagation.get_neural_network();

    const Index parameters_number = neural_network->get_parameters_buffer_size();

    float* const velocity_ptr = momentum > 0.0f
        ? optimizer_data.views[StochasticGradientDescent::Velocity].as<float>()
        : nullptr;

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

OPENNN_CUDA_STUB(void, update_parameters_cuda,
                 (BackPropagation&, OptimizerData&,
                  float, float, bool))

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

void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  OptimizerData& optimizer_data,
                                                  UpdateMode mode)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    if (mode == UpdateMode::Capturable)
    {
#ifdef OPENNN_HAS_CUDA
        clip_gradient_norm(back_propagation, gradient_clip_norm);

        float* const velocity_ptr = momentum > 0.0f
            ? optimizer_data.views[Velocity].as<float>()
            : nullptr;

        return sgd_update_capturable_cuda(
                   neural_network->get_parameters_buffer_size(),
                   neural_network->get_parameters_data(),
                   velocity_ptr,
                   back_propagation.gradient.as<float>(),
                   optimizer_data.views[GraphLearningRate].as<float>(),
                   momentum,
                   nesterov,
                   neural_network->get_parameters_bf16_mirror_data(),
                   device::get_compute_stream());
#else
        throw runtime_error("Capturable SGD parameter updates require CUDA support.");
#endif
    }

    const float current_learning_rate = optimizer_data.current_learning_rate;
    if (current_learning_rate == 0.0f)
        return;

    throw_if(momentum > 0.0f && optimizer_data.views.empty(),
             "StochasticGradientDescent::update_parameters: velocity buffer is not initialized.");

    clip_gradient_norm(back_propagation, gradient_clip_norm);

    if (neural_network->is_gpu())
        return update_parameters_cuda(back_propagation, optimizer_data,
                                      current_learning_rate, momentum, nesterov);

    VectorMap parameters = neural_network->get_parameters_map();

    VectorMap gradient = back_propagation.gradient.as_vector();

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

void StochasticGradientDescent::setup_optimizer_data(OptimizerData& optimizer_data,
                                                     Index parameters_number,
                                                     Device device)
{
    const bool use_graph = can_use_cuda_graph();
    if (momentum > 0.0f || use_graph)
        optimizer_data.set({momentum > 0.0f ? Shape{parameters_number} : Shape{},
                            use_graph ? Shape{1} : Shape{}}, device);

    optimizer_data.current_learning_rate = initial_learning_rate;

}

void StochasticGradientDescent::on_epoch_begin(Index epoch, OptimizerData& optimizer_data)
{
    optimizer_data.current_learning_rate =
        initial_learning_rate / (1.0f + float(epoch) * initial_decay);

#ifdef OPENNN_HAS_CUDA
    if (can_use_cuda_graph())
        set_scalar_device_cuda(optimizer_data.views[GraphLearningRate].as<float>(),
                               optimizer_data.current_learning_rate,
                               device::get_compute_stream());
#endif
}

void StochasticGradientDescent::to_JSON(JsonWriter& printer) const
{
    printer.open_element("StochasticGradientDescent");

    write_json(printer, {
        {"BatchSize", batch_size},
        {"InitialLearningRate", initial_learning_rate},
        {"InitialDecay", initial_decay},
        {"Momentum", momentum},
        {"Nesterov", nesterov},
        {"ApplyMomentum", momentum > 0.0f}
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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
