//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/dataset.h"
#include "opennn/training_strategy/error_functions.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/profiler.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/core/cuda/kernel_optimizers.cuh"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void update_parameters_cuda(NeuralNetwork* neural_network,
                                   BackPropagation& back_propagation,
                                   OptimizerData& optimization_data,
                                   float beta_1,
                                   float beta_2,
                                   float learning_rate,
                                   float bias_correction_1,
                                   float bias_correction_2)
{
    PROFILE_SCOPE("optim:adam_update_cuda");
    const Index parameters_number = neural_network->get_parameters_buffer_size();

    adam_update_cuda(
        parameters_number,
        neural_network->get_parameters_data(),
        optimization_data.views[AdaptiveMomentEstimation::GradientMoment].as<float>(),
        optimization_data.views[AdaptiveMomentEstimation::SquareGradientMoment].as<float>(),
        back_propagation.gradient.as<float>(),
        beta_1,
        beta_2,
        learning_rate,
        EPSILON,
        bias_correction_1,
        bias_correction_2,
        neural_network->get_parameters_bf16_mirror_data());
}

#else

OPENNN_CUDA_STUB(void, update_parameters_cuda,
                 (NeuralNetwork*, BackPropagation&, OptimizerData&,
                  float, float, float, float, float))

#endif

static void accumulate_scaled_gradient(Buffer& accumulator, const Buffer& gradient, float alpha)
{
#ifdef OPENNN_HAS_CUDA
    if (accumulator.device_type == Device::CUDA)
    {
        CHECK_CUBLAS(cublasSaxpy(Backend::get_cublas_handle(),
                                 int(gradient.size_in_floats()), &alpha,
                                 gradient.as<float>(), 1,
                                 accumulator.as<float>(), 1));
        return;
    }
#endif
    accumulator.as_vector().noalias() += alpha * gradient.as_vector();
}

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

void AdaptiveMomentEstimation::setup_optimizer_data(OptimizerData& optimization_data,
                                                    Index parameters_number,
                                                    Device device)
{
    const bool use_graph = can_use_cuda_graph();
    throw_if(update_period > 1 && use_graph,
             "gradient accumulation is not supported with the CUDA graph.");

    optimization_data.set({Shape{parameters_number},
                           Shape{parameters_number},
                           use_graph ? Shape{3} : Shape{}}, device);
    optimization_data.iteration = 0;
    optimization_data.accumulated_batches = 0;
    if (update_period > 1)
    {
        optimization_data.gradient_accumulator.resize_bytes(
            parameters_number * Index(sizeof(float)), device);
        optimization_data.gradient_accumulator.setZero();
    }

}

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data,
                                                 UpdateMode mode)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    if (mode == UpdateMode::Capturable)
    {
#ifdef OPENNN_HAS_CUDA
        clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);

        float* const graph_scalars = optimization_data.views[GraphScalars].as<float>();
        int* const graph_step = reinterpret_cast<int*>(graph_scalars);
        float* const graph_learning_rate = graph_scalars + 1;
        float* const graph_epsilon = graph_scalars + 2;

        adam_update_capturable_cuda(
            neural_network->get_parameters_buffer_size(),
            neural_network->get_parameters_data(),
            optimization_data.views[GradientMoment].as<float>(),
            optimization_data.views[SquareGradientMoment].as<float>(),
            back_propagation.gradient.as<float>(),
            beta_1, beta_2, learning_rate, EPSILON,
            graph_step,
            graph_learning_rate,
            graph_epsilon,
            neural_network->get_parameters_bf16_mirror_data(),
            Backend::get_compute_stream());
        return;
#else
        throw runtime_error("Capturable Adam parameter updates require CUDA support.");
#endif
    }

    const Index period = max(Index(1), update_period);

    if (period > 1)
    {
        accumulate_scaled_gradient(optimization_data.gradient_accumulator,
                                   back_propagation.gradient,
                                   1.0f / float(period));

        if (++optimization_data.accumulated_batches < period) return;

        Buffer& accumulator = optimization_data.gradient_accumulator;
        device::copy_async(back_propagation.gradient.data, accumulator.data,
                           accumulator.bytes,
                           accumulator.device_type, accumulator.device_type,
                           accumulator.device_type == Device::CUDA ? Backend::get_compute_stream() : nullptr);
        accumulator.setZero();
        optimization_data.accumulated_batches = 0;
    }

    optimization_data.iteration++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);
    }

    const float iteration = static_cast<float>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - pow(beta_1, iteration);
    const float bias_correction_2 = 1.0f - pow(beta_2, iteration);

    if (neural_network->is_gpu())
    {
        update_parameters_cuda(neural_network, back_propagation, optimization_data,
                               beta_1, beta_2, learning_rate,
                               bias_correction_1, bias_correction_2);
        return;
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

    #pragma omp parallel for if(parameters_size > 4096)
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
    if (root_element->has("LearningRate"))     set_learning_rate(read_json_float(root_element, "LearningRate"));
    if (root_element->has("Beta1"))            set_beta_1(read_json_float(root_element, "Beta1"));
    if (root_element->has("Beta2"))            set_beta_2(read_json_float(root_element, "Beta2"));
    read_common_json(root_element);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
