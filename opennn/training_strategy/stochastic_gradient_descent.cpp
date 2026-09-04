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
#include "opennn/training_strategy/kernel_optimizers.cuh"
#include "opennn/training_strategy/loss.h"

namespace opennn
{

StochasticGradientDescent::StochasticGradientDescent(Loss* new_loss)
    : Optimizer(new_loss)
{
    name = "StochasticGradientDescent";
    maximum_time = 3600.0f;
    maximum_epochs = 1000;
    display_period = 100;
}

void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  OptimizerData& optimizer_data,
                                                  UpdateMode mode)
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    const vector<BackPropagation::GradientSlice>& gradient_slices =
        back_propagation.get_gradient_slices();

    if (mode == UpdateMode::Capturable)
    {
#ifdef OPENNN_HAS_CUDA
        clip_gradient_norm(back_propagation, gradient_clip_norm);

        float* const velocity_ptr = momentum > 0.0f
            ? optimizer_data.views[Velocity].as<float>()
            : nullptr;
        float* const parameters = neural_network->get_parameters_data();
        bfloat16* const mirror =
            neural_network->get_parameters_bf16_mirror_data();
        cudaStream_t stream = device::get_compute_stream();

        for(const BackPropagation::GradientSlice& slice : gradient_slices)
        {
            const Index offset = slice.parameter_offset;
            sgd_update_capturable_cuda(
                slice.values.size(),
                parameters + offset,
                velocity_ptr ? velocity_ptr + offset : nullptr,
                slice.values.as<float>(),
                optimizer_data.views[GraphLearningRate].as<float>(),
                momentum,
                nesterov,
                mirror ? mirror + offset : nullptr,
                stream);
        }
        return;
#else
        throw runtime_error("Capturable SGD parameter updates require CUDA support.");
#endif
    }

    if (current_learning_rate == 0.0f)
        return;

    throw_if(momentum > 0.0f && optimizer_data.views.empty(),
             "StochasticGradientDescent::update_parameters: velocity buffer is not initialized.");

    clip_gradient_norm(back_propagation, gradient_clip_norm);

    if (neural_network->is_gpu())
    {
#ifdef OPENNN_HAS_CUDA
        float* const velocity_ptr = momentum > 0.0f
            ? optimizer_data.views[Velocity].as<float>()
            : nullptr;
        float* const parameters = neural_network->get_parameters_data();
        bfloat16* const mirror =
            neural_network->get_parameters_bf16_mirror_data();

        PROFILE_SCOPE("optim:sgd_update_cuda");

        for(const BackPropagation::GradientSlice& slice : gradient_slices)
        {
            const Index offset = slice.parameter_offset;
            sgd_update_cuda(
                slice.values.size(),
                parameters + offset,
                velocity_ptr ? velocity_ptr + offset : nullptr,
                slice.values.as<float>(),
                current_learning_rate, momentum, nesterov,
                mirror ? mirror + offset : nullptr);
        }
        return;
#else
        throw runtime_error("SGD parameter updates on GPU require CUDA support.");
#endif
    }

    VectorMap parameters = neural_network->get_parameters_map();

    float* const velocity = momentum > 0.0f
        ? optimizer_data.views[Velocity].as<float>()
        : nullptr;

    const auto update_range = [&](const Index offset,
                                  const VectorMap& gradient)
    {
        const Index range_size = gradient.size();

        if (momentum <= 0.0f)
        {
            #pragma omp parallel for if(range_size > 65536)
            for (Index i = 0; i < range_size; ++i)
                parameters(offset + i) -= current_learning_rate * gradient(i);
        }
        else
        {
            #pragma omp parallel for if(range_size > 65536)
            for (Index i = 0; i < range_size; ++i)
            {
                const Index parameter = offset + i;
                const float learning_rate_gradient =
                    current_learning_rate * gradient(i);
                const float new_velocity =
                    momentum * velocity[parameter] - learning_rate_gradient;
                velocity[parameter] = new_velocity;
                parameters(parameter) += nesterov
                    ? momentum * new_velocity - learning_rate_gradient
                    : new_velocity;
            }
        }
    };

    for(const BackPropagation::GradientSlice& slice : gradient_slices)
    {
        update_range(slice.parameter_offset,
                     slice.values.as_vector());
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

    current_learning_rate = initial_learning_rate;

}

void StochasticGradientDescent::on_epoch_begin(Index epoch, OptimizerData& optimizer_data)
{
    current_learning_rate = initial_learning_rate / (1.0f + float(epoch) * initial_decay);

#ifdef OPENNN_HAS_CUDA
    if (can_use_cuda_graph())
        set_scalar_device_cuda(optimizer_data.views[GraphLearningRate].as<float>(),
                               current_learning_rate,
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

    set_initial_learning_rate(read_json_float(root_element, "InitialLearningRate", initial_learning_rate));
    set_initial_decay(read_json_float(root_element, "InitialDecay", initial_decay));
    set_nesterov(read_json_bool(root_element, "Nesterov", nesterov));

    set_momentum(read_json_float(root_element, "Momentum",
                                 read_json_bool(root_element, "ApplyMomentum") ? 0.9f : 0.0f));

    read_common_json(root_element);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
