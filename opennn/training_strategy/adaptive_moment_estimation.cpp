//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/adaptive_moment_estimation.h"

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/training_strategy/error_functions.h"
#include "opennn/training_strategy/kernel_optimizers.cuh"
#include "opennn/training_strategy/loss.h"

namespace opennn
{

AdaptiveMomentEstimation::AdaptiveMomentEstimation(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void AdaptiveMomentEstimation::set_beta_1(const float new_beta_1)
{
    throw_if(new_beta_1 < 0.0f || new_beta_1 >= 1.0f,
             "AdaptiveMomentEstimation::set_beta_1: beta_1 must be in [0, 1).");

    beta_1 = new_beta_1;
}

void AdaptiveMomentEstimation::set_beta_2(const float new_beta_2)
{
    throw_if(new_beta_2 < 0.0f || new_beta_2 >= 1.0f,
             "AdaptiveMomentEstimation::set_beta_2: beta_2 must be in [0, 1).");

    beta_2 = new_beta_2;
}

void AdaptiveMomentEstimation::set_default()
{
    batch_size = 0;
    display_period = 100;
    name = "AdaptiveMomentEstimation";
}

void AdaptiveMomentEstimation::configure_for_task(NetworkTask task)
{
    static constexpr float language_model_learning_rate = 0.0001f;

    Optimizer::configure_for_task(task);

    if (task == NetworkTask::LanguageModeling)
        learning_rate = language_model_learning_rate;
}

void AdaptiveMomentEstimation::setup_optimizer_data(OptimizerData& optimization_data,
                                                    Index parameters_number,
                                                    Device device)
{
    const bool use_graph = can_use_cuda_graph();

    optimization_data.set({Shape{parameters_number},
                           Shape{parameters_number},
                           use_graph ? Shape{4} : Shape{}}, device);
    optimization_data.iteration = 0;

#ifdef OPENNN_HAS_CUDA
    if (use_graph)
        set_scalar_device_cuda(optimization_data.views[GraphScalars].as<float>() + 3,
                               learning_rate,
                               device::get_compute_stream());
#endif
}

void AdaptiveMomentEstimation::on_epoch_begin(Index, OptimizerData& optimization_data)
{
#ifdef OPENNN_HAS_CUDA
    if (can_use_cuda_graph() && optimization_data.views[GraphScalars].size() >= 4)
        set_scalar_device_cuda(optimization_data.views[GraphScalars].as<float>() + 3,
                               learning_rate,
                               device::get_compute_stream());
#else
    (void)optimization_data;
#endif
}

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data,
                                                 UpdateMode mode)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    const bool has_graph_scalars =
        optimization_data.views.size() > size_t(GraphScalars)
        && optimization_data.views[GraphScalars].size() >= 4;

    if (mode == UpdateMode::Capturable
        || (has_graph_scalars && neural_network->is_gpu() && can_use_cuda_graph()))
    {
#ifdef OPENNN_HAS_CUDA
        clip_gradient_norm(back_propagation, gradient_clip_norm);

        float* const graph_scalars = optimization_data.views[GraphScalars].as<float>();
        int* const graph_step = reinterpret_cast<int*>(graph_scalars);
        float* const graph_learning_rate = graph_scalars + 1;
        float* const graph_epsilon = graph_scalars + 2;
        const float* const graph_base_learning_rate = graph_scalars + 3;

        return adam_update_capturable_cuda(
                   neural_network->get_parameters_buffer_size(),
                   neural_network->get_parameters_data(),
                   optimization_data.views[GradientMoment].as<float>(),
                   optimization_data.views[SquareGradientMoment].as<float>(),
                   back_propagation.gradient.as<float>(),
                   beta_1, beta_2, graph_base_learning_rate, EPSILON,
                   graph_step,
                   graph_learning_rate,
                   graph_epsilon,
                   neural_network->get_parameters_bf16_mirror_data(),
                   device::get_compute_stream());
#else
        throw runtime_error("Capturable Adam parameter updates require CUDA support.");
#endif
    }

    optimization_data.iteration++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation, gradient_clip_norm);
    }

    const float iteration = static_cast<float>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - pow(beta_1, iteration);
    const float bias_correction_2 = 1.0f - pow(beta_2, iteration);

    if (neural_network->is_gpu())
    {
#ifdef OPENNN_HAS_CUDA
        PROFILE_SCOPE("optim:adam_update_cuda");

        return adam_update_cuda(
                   neural_network->get_parameters_buffer_size(),
                   neural_network->get_parameters_data(),
                   optimization_data.views[GradientMoment].as<float>(),
                   optimization_data.views[SquareGradientMoment].as<float>(),
                   back_propagation.gradient.as<float>(),
                   beta_1, beta_2, learning_rate, EPSILON,
                   bias_correction_1, bias_correction_2,
                   neural_network->get_parameters_bf16_mirror_data());
#else
        throw runtime_error("Adam parameter updates on GPU require CUDA support.");
#endif
    }

    VectorMap parameters = neural_network->get_parameters_map();

    VectorMap gradient_exponential_decay = optimization_data.views[GradientMoment].as_vector();
    VectorMap square_gradient_exponential_decay = optimization_data.views[SquareGradientMoment].as_vector();

    VectorMap gradient = back_propagation.gradient.as_vector();

    const Index parameters_size = parameters.size();
    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const float sqrt_bias_correction_2 = sqrt(bias_correction_2);
    const float effective_learning_rate = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_epsilon = EPSILON * sqrt_bias_correction_2;

    {
        PROFILE_SCOPE_HOST("optim:adam_update_cpu");

        #pragma omp parallel for if(parameters_size > 65536)
        for (Index i = 0; i < parameters_size; ++i)
        {
            const float gradient_value = gradient(i);

            auto& first_moment = gradient_exponential_decay(i);
            auto& second_moment = square_gradient_exponential_decay(i);

            first_moment = beta_1 * first_moment + one_minus_beta_1 * gradient_value;
            second_moment = beta_2 * second_moment + one_minus_beta_2 * gradient_value * gradient_value;

            parameters(i) -= effective_learning_rate * first_moment / (sqrt(second_moment) + effective_epsilon);
        }
    }
}

void AdaptiveMomentEstimation::to_JSON(JsonWriter& printer) const
{
    printer.open_element("AdaptiveMomentEstimation");

    add_json_field(printer, "BatchSize", batch_size);
    add_json_field(printer, "LearningRate", learning_rate);
    add_json_field(printer, "Beta1", beta_1);
    add_json_field(printer, "Beta2", beta_2);
    write_common_json(printer);

    printer.close_element();
}

void AdaptiveMomentEstimation::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_json_index(root_element, "BatchSize"));
    set_learning_rate(read_json_float(root_element, "LearningRate", learning_rate));
    set_beta_1(read_json_float(root_element, "Beta1", beta_1));
    set_beta_2(read_json_float(root_element, "Beta2", beta_2));
    read_common_json(root_element);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
